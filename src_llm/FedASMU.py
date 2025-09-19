#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10+
#
# FedASMU-FA for sLM (server-side only, minimal patch over BRAIN-LLM)
# - Eq.(1) aggregation, Eq.(2) alpha(+ staleness bound), Eq.(3) dynamic control update
# - Uses the *previous aggregation* version indices (o) in server gradients
# - No device-side mixing (FA)
#
# NOTE: This follows the same structure/pattern as the CNN version you provided.
#       No extra optimizations (e.g., streaming inner products) are introduced.

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import gc
import torch  # for state_dict vectorization
from tensorboardX import SummaryWriter

from src_llm.impls.options import args_parser
from src_llm.impls.update import LocalUpdate, ByzantineLocalUpdate, test_inference
from src_llm.impls.cache import ItemCache
from src_llm.impls.utils import pareto_splits, uniform_splits, compose_weight

from src_llm.llama.model import make_net


# ---------- helpers: vectorize state_dict & inner product (same as CNN style) ----------
def _sd_to_vec(sd):
    vecs = []
    for k, v in sd.items():
        if torch.is_tensor(v):
            vecs.append(
                v.detach().cpu().float().reshape(-1).numpy()  # view -> reshape (non-contig safe)
            )
    if len(vecs) == 0:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate(vecs, axis=0)


def _vec_ip(a, b):
    n = min(a.size, b.size)
    return float(np.dot(a[:n], b[:n]))


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    os.makedirs('./logs_llm', exist_ok=True)
    logger = SummaryWriter('./logs_llm')

    args = args_parser()

    # -------------------- FedASMU server hyperparams (same default pattern as CNN) --------------------
    num_byzantines = int(getattr(args, 'byzantines', 0))

    # Eq.(2) & Alg.1 params
    args.mu_alpha = getattr(args, 'mu_alpha', 1.0)
    args.lam0 = getattr(args, 'lam0',     1.0)
    args.sig0 = getattr(args, 'sig0',     1.0)
    args.iota0 = getattr(args, 'iota0',    0.0)
    # staleness bound
    args.tau = getattr(args, 'tau', getattr(args, 'stale', 99))
    # Eq.(3) learning rates (Appendix server grads)
    args.eta_lam = getattr(args, 'eta_lam',  1e-3)
    args.eta_sig = getattr(args, 'eta_sig',  1e-3)
    args.eta_iota = getattr(args, 'eta_iota', 1e-3)
    # local optimizer LR used in Appendix gradient approx
    args_lr_for_appendix = getattr(args, 'lr', 0.03)
    local_L = getattr(args, 'local_ep', 1)

    # -------------------- dataset splits (same as original BRAIN-LLM driver) --------------------
    os.makedirs('./save_llm', exist_ok=True)
    os.makedirs('./save_llm/objects', exist_ok=True)

    if getattr(args, 'iid', True):
        user_groups = uniform_splits(
            36718,  # TODO: wikitext2 (args)
            args.num_users - args.byzantines
        )
    else:
        user_groups = pareto_splits(
            36718,  # TODO: wikitext2 (args)
            args.num_users - args.byzantines,
            alpha=3.0,
            min_per_label=128,
            seed=int(time.time() * 1000) & (2**32 - 1)
        )

    # -------------------- build model/tokenizer --------------------
    model_name = getattr(args, 'model_name', "HuggingFaceTB/SmolLM2-135M")
    global_model, tokenizer = make_net(model_name=model_name)
    global_model.train()
    print(global_model)

    # copy weights + global version (t)
    global_weights = global_model.state_dict()
    global_version = 0  # t in the paper

    # ---------- per-client control params (λ,σ,ι) & history for Appendix grads ----------
    ctrl = {i: {'lam': args.lam0, 'sig': args.sig0, 'iota': args.iota0}
            for i in range(args.num_users)}
    hist = {i: {'prev_local': None, 'prev_before': None, 'prev_o': None, 'prev_agg_version': None}
            for i in range(args.num_users)}

    # Metrics
    ppl_collect = []

    # Cache (staleness simulation), same API/usage style
    cache = ItemCache(min_counter=0, max_counter=args.stale)

    # -------------- small utils for FedASMU Eq.(2), Appendix grads for Eq.(3) ----------
    def _alpha_and_tau(t, o, c, mu_alpha):
        """Eq.(2): compute alpha and staleness tau = t - o + 1."""
        t_eff = max(1, int(t))
        tau = max(1, t_eff - int(o) + 1)
        lam = float(c['lam'])
        sig = max(0.0, float(c['sig']))
        iota = float(c['iota'])
        xi = (lam * np.sqrt(t_eff)) / \
            (t_eff * (tau ** max(sig, 0.0)) + 1e-12) + iota
        xi = max(0.0, xi)  # keep non-negative
        alpha = (mu_alpha * xi) / (1.0 + mu_alpha * xi + 1e-12)
        return float(np.clip(alpha, 0.0, 1.0)), tau

    def _server_ctrl_grads(i, o, wi_o, w_base_o,
                           prev_local, prev_before, prev_o, prev_agg_version,
                           mu_alpha, lam, sig, iota,
                           eta_i, L):
        """
        Appendix server grads (use *previous aggregation* version indices):
          - (w_i^o - w_o) · (w_i^{o'} - w_{o-1})
          - sqrt{o-1}, (o - o')^σ, and (1 + μ_α ξ_{o-1})^2 in the denominator
        """
        if (prev_local is None) or (prev_before is None) or (prev_o is None) or (prev_agg_version is None):
            return 0.0, 0.0, 0.0

        # vectors
        v1 = _sd_to_vec(wi_o) - _sd_to_vec(w_base_o)           # (w_i^o - w_o)
        # (w_i^{o'} - w_{o-1})
        v2 = _sd_to_vec(prev_local) - _sd_to_vec(prev_before)
        ip = _vec_ip(v1, v2)

        # quantities at previous aggregation 'o'
        # previous aggregation result version = o
        o_prev = int(prev_agg_version)
        o_minus_1 = max(1, o_prev - 1)
        tau_prev = max(1, o_prev - int(prev_o))  # (o - o')

        xi_prev = (lam * np.sqrt(o_minus_1)) / (o_minus_1 *
                                                (tau_prev ** max(sig, 0.0)) + 1e-12) + iota
        denom_sq = (1.0 + mu_alpha * xi_prev + 1e-12) ** 2

        etaL = max(1e-12, float(eta_i) * max(1, int(L)))
        # grads (consistent signs; ∂/∂σ of tau^{-σ} = -ln(tau) * tau^{-σ})
        g_lam = (mu_alpha * ip) / (etaL * np.sqrt(o_minus_1)
                                   * denom_sq * (tau_prev ** max(sig, 0.0)))
        ln_tau = np.log(max(1.0, float(tau_prev)))
        g_sig = (mu_alpha * ln_tau * (-ip)) / (etaL * np.sqrt(o_minus_1)
                                               * denom_sq * (tau_prev ** max(sig, 0.0)))
        g_iota = (mu_alpha * ip) / (etaL * denom_sq)

        return float(g_lam), float(g_sig), float(g_iota)

    # ------------------------------- main loop -------------------------------
    for epoch in range(args.epochs + args.stale):
        print("[INFO]", epoch)

        if (len(cache.cache) == 0) and (epoch >= args.epochs):
            break

        # try:
        arrived_items = []

        if (epoch < args.epochs):
            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(
                range(args.num_users), m, replace=False)

            for idx in idxs_users:
                # record original version o when this client starts training
                o_version = int(global_version)
                if idx >= args.byzantines:
                    local_model = LocalUpdate(args=args, tokenizer=tokenizer)
                    (skip_num, take_num) = user_groups[idx - num_byzantines]
                    local_model.set_dataset_train(
                        skip_num, take_num, seed=int(
                            time.time() * 1000) & (2**32 - 1)
                    )
                else:
                    local_model = ByzantineLocalUpdate(
                        args=args, tokenizer=tokenizer)

                # keep driver semantics for logging/schedulers
                w, avg_train_loss = local_model.update_weights(
                    model=copy.deepcopy(global_model),
                    epochs=args.local_ep,
                    global_round=epoch
                )

                # store (update, original version o, client i, and the base global w_o) into cache
                cache.add_item_with_random_counter({
                    'w': copy.deepcopy(w),                 # local upload w_i^o
                    'o': o_version,                        # original global version o
                    'i': int(idx),                         # client id
                    'base': copy.deepcopy(global_weights)  # w_o sent to client
                })

        # at epoch end, drain whatever arrived (single drain per epoch for simplicity)
        arrived_items.extend(cache.update_counters())

        # ---------------- FedASMU server aggregation (Alg.1 lines 8–14) ----------------
        if len(arrived_items) != 0:
            for item in arrived_items:
                i = int(item['i'])
                o = int(item['o'])
                w_i_o = item['w']
                w_base = item['base']  # w_o (global when client started)

                # (9–10) staleness bound check
                _alpha_tmp, tau_now = _alpha_and_tau(
                    global_version, o, ctrl[i], args.mu_alpha)
                if tau_now > int(args.tau):
                    continue  # discard

                # (12) dynamic update of (λ,σ,ι) via Appendix grads (Eq.(3))
                lam_i = ctrl[i]['lam']
                sig_i = ctrl[i]['sig']
                iota_i = ctrl[i]['iota']

                # Gate: only update controls when this upload started from the *previous aggregation* version
                do_ctrl_update = (
                    hist[i]['prev_local'] is not None and
                    hist[i]['prev_before'] is not None and
                    hist[i]['prev_o'] is not None and
                    hist[i]['prev_agg_version'] is not None and
                    (o == int(hist[i]['prev_agg_version']))
                )
                if do_ctrl_update:
                    g_lam, g_sig, g_iota = _server_ctrl_grads(
                        i=i, o=o, wi_o=w_i_o, w_base_o=w_base,
                        prev_local=hist[i]['prev_local'],
                        prev_before=hist[i]['prev_before'],
                        prev_o=hist[i]['prev_o'],
                        prev_agg_version=hist[i]['prev_agg_version'],
                        mu_alpha=args.mu_alpha,
                        lam=lam_i, sig=sig_i, iota=iota_i,
                        eta_i=args_lr_for_appendix, L=local_L
                    )
                    # SGD steps for control params (Eq.(3)); enforce σ ≥ 0
                    ctrl[i]['lam'] = float(
                        ctrl[i]['lam'] - args.eta_lam * g_lam)
                    ctrl[i]['sig'] = float(
                        max(0.0, ctrl[i]['sig'] - args.eta_sig * g_sig))
                    ctrl[i]['iota'] = float(
                        ctrl[i]['iota'] - args.eta_iota * g_iota)

                # (13) compute alpha with updated controls (Eq.(2))
                alpha, tau_now = _alpha_and_tau(
                    global_version, o, ctrl[i], args.mu_alpha)

                # (14) update global model (Eq.(1))
                # w_{o-1} before this aggregation
                pre_agg_w = copy.deepcopy(global_weights)
                global_weights = compose_weight(global_weights, w_i_o, alpha)
                global_model.load_state_dict(global_weights)
                # aggregation result version == o (for this event)
                global_version += 1

                # update history for this client (used next time in Appendix grads)
                hist[i]['prev_local'] = w_i_o             # w_i^{o'}
                hist[i]['prev_before'] = pre_agg_w         # w_{o-1}
                hist[i]['prev_o'] = o                 # o'
                # this aggregation result version o
                hist[i]['prev_agg_version'] = int(global_version)

        # ---------------- evaluation (perplexity) ----------------
        ppl, ppl_std = test_inference(args, global_model, tokenizer)
        print(f"[INFO] Wikitext PPL (wikitext2): {ppl} +- {ppl_std}")
        ppl_collect.append(ppl)

        gc.collect()
        torch.cuda.empty_cache()

        # except RuntimeError as e:  # TODO: print error msg
        #     # if 'out of memory' in str(e):
        #     print(
        #         f"[WARNING] CUDA OOM at epoch {epoch}, stopping training loop.")
        #     torch.cuda.empty_cache()
        #     args.epochs = epoch
        #     break  # or continue
        #     # else:
        #     # raise

    # save results
    file_name = './save_llm/objects/fedasmu_fa_server_{}_C{}_iid{}_E{}_B{}_Z{}_S{}_tau{}_muA{}_{}.pkl'.\
        format(args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.byzantines, args.stale,
               args.tau, args.mu_alpha, time.time())

    with open(file_name, 'wb') as f:
        pickle.dump([ppl_collect], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # plotting (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    plt.figure()
    plt.title('FedASMU-FA(Server, LLM): PPL vs Communication rounds')
    plt.plot(range(len(ppl_collect)), ppl_collect)
    plt.ylabel('Test PPL (lower is better)')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save_llm/fedasmu_fa_llm_ppl.png')
