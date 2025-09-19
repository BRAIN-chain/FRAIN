#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#
# FedASMU-FA (server-side only, minimal patch over BRAIN driver)
# Implements: Eq.(1) aggregation, Eq.(2) alpha, tau-bound discard (Alg.1 L9–L10),
# and Eq.(3) dynamic control update using Appendix server gradients with the
# correct version indices (use the *previous aggregation* version o).  (Paper p.4–5, p.11)
#
# Disabled by design in FA: Algorithm 2 (device-side fresh-global mixing: Eq.(7)–(9)).
#
# References: FedASMU (AAAI'24). See paper Eq.(1)–(3), Algorithm 1, Appendix (server part).
# (We keep impls/* and airbench/* modules intact.)

import os
import copy
import time
import pickle
import numpy as np
import csv
from tqdm import tqdm

import torch  # for state_dict vectorization
from tensorboardX import SummaryWriter

from impls.options import args_parser
from impls.update import LocalUpdate, ByzantineLocalUpdate, test_inference
from impls.utils import get_dataset, compose_weight, exp_details

from impls.cache import ItemCache

from airbench.model import make_net
from airbench.hyperparameters import hyp


# ---------- helpers: vectorize state_dict & inner product ----------
def _sd_to_vec(sd):
    vecs = []
    for k, v in sd.items():
        if torch.is_tensor(v):
            vecs.append(
                # v.detach().cpu().float().view(-1).numpy()
                v.detach().cpu().float().reshape(-1).numpy()
            )
    if len(vecs) == 0:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate(vecs, axis=0)


def _vec_ip(a, b):
    n = min(a.size, b.size)
    return float(np.dot(a[:n], b[:n]))


if __name__ == '__main__':
    start_time = time.time()
    traning_times = []

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    num_byzantines = int(getattr(args, 'byzantines', 0))

    # -------------------- FedASMU server hyperparams (safe defaults) --------------------
    # Eq.(2) & Alg.1 params
    args.mu_alpha = getattr(args, 'mu_alpha', 1.0)
    args.lam0 = getattr(args, 'lam0', 1.0)
    args.sig0 = getattr(args, 'sig0', 1.0)
    args.iota0 = getattr(args, 'iota0', 0.0)
    # staleness bound
    args.tau = getattr(args, 'tau', getattr(args, 'stale', 99))
    # Eq.(3) learning rates (Appendix server grads)
    args.eta_lam = getattr(args, 'eta_lam',  1e-3)
    args.eta_sig = getattr(args, 'eta_sig',  1e-3)
    args.eta_iota = getattr(args, 'eta_iota', 1e-3)
    args_lr_for_appendix = getattr(args, 'lr', 0.03)
    local_L = getattr(args, 'local_ep', 1)

    # -------------------- load dataset and user groups --------------------
    os.makedirs('./save', exist_ok=True)
    os.makedirs('./save/objects', exist_ok=True)
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # -------------------- BUILD MODEL --------------------
    if (args.model == 'cnn') and (args.dataset == 'cifar'):
        pass
    else:
        exit('Error: unrecognized model')
    widths = hyp['net']['widths']
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    scaling_factor = hyp['net']['scaling_factor']
    global_model = make_net(widths, batchnorm_momentum, scaling_factor)
    global_model.train()
    print(global_model)

    # copy weights + global version (t)
    global_weights = global_model.state_dict()
    global_version = 0  # t in the paper

    # ---------- per-client control params (λ,σ,ι) & history for Appendix grads ----------
    ctrl = {i: {'lam': args.lam0, 'sig': args.sig0, 'iota': args.iota0}
            for i in range(args.num_users)}
    # for server-side gradient approximation (Appendix "Details on the Server", p.11)
    hist = {i: {'prev_local': None, 'prev_before': None, 'prev_o': None, 'prev_agg_version': None}
            for i in range(args.num_users)}

    # Training
    test_loss_collect, test_acc_collect = [], []

    # Cache (staleness simulation kept as-is)
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
        Appendix 'Details on the Server' scalar grads for Eq.(3) (paper p.11).
        Uses the *previous aggregation* version indices:
          - inner products: (w_i^o - w_o) · (w_i^{o'} - w_{o-1})
          - denominators: sqrt{o-1}, (o - o')^{sigma}, and (1 + mu_alpha * xi_{o-1})^2
        """
        if (prev_local is None) or (prev_before is None) or (prev_o is None) or (prev_agg_version is None):
            return 0.0, 0.0, 0.0

        # vectors
        v1 = _sd_to_vec(wi_o) - _sd_to_vec(w_base_o)          # (w_i^o - w_o)
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
        # grads (consistent signs; see Appendix p.11; d/dsigma brings -ln(tau))
        g_lam = (mu_alpha * ip) / (etaL * np.sqrt(o_minus_1)
                                   * denom_sq * (tau_prev ** max(sig, 0.0)))
        ln_tau = np.log(max(1.0, float(tau_prev)))
        g_sig = (mu_alpha * ln_tau * (-ip)) / (etaL * np.sqrt(o_minus_1)
                                               * denom_sq * (tau_prev ** max(sig, 0.0)))
        g_iota = (mu_alpha * ip) / (etaL * denom_sq)

        return float(g_lam), float(g_sig), float(g_iota)

    # ------------------------------- main loop -------------------------------
    for epoch in tqdm(range(args.epochs + args.stale)):
        if (len(cache.cache) == 0) and (epoch >= args.epochs):
            break

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
                    local_model = LocalUpdate(args=args, hyps=hyp,
                                              dataset=train_dataset,
                                              idxs=user_groups[idx -
                                                               num_byzantines],
                                              logger=logger)
                    traning_start = time.time()
                else:
                    local_model = ByzantineLocalUpdate(args=args, hyps=None,
                                                       dataset=train_dataset, idxs=[],
                                                       logger=logger)

                # IMPORTANT: keep original driver semantics for logging/schedulers
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), epochs=args.local_ep, global_round=epoch)
                if idx >= args.byzantines:
                    traning_times.append(time.time() - traning_start)

                # store (update, original version o, client i, and the base global w_o) into cache
                cache.add_item_with_random_counter({
                    'w': copy.deepcopy(w),             # local upload w_i^o
                    'o': o_version,                    # original global version o
                    'i': int(idx),                     # client id
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
                alpha_tmp, tau_now = _alpha_and_tau(
                    global_version, o, ctrl[i], args.mu_alpha)
                if tau_now > int(args.tau):
                    # discard this upload
                    continue

                # (12) dynamic update of (λ,σ,ι) via Appendix grads (Eq.(3))
                # gate: only update controls when current upload started from the *previous aggregation* version
                lam_i = ctrl[i]['lam']
                sig_i = ctrl[i]['sig']
                iota_i = ctrl[i]['iota']
                do_ctrl_update = False
                if (hist[i]['prev_local'] is not None and
                    hist[i]['prev_before'] is not None and
                    hist[i]['prev_o'] is not None and
                        hist[i]['prev_agg_version'] is not None):
                    if o == int(hist[i]['prev_agg_version']):
                        do_ctrl_update = True

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
                    # SGD steps for control params (Eq.(3))
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

        # ---------------- evaluation ----------------
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_collect.append(test_acc)
        test_loss_collect.append(test_loss)

    # save results
    file_name = './save/objects/fedasmu_fa_server_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_S{}_tau{}_muA{}_{}.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.byzantines, args.stale,
               args.tau, args.mu_alpha, time.time())

    with open(file_name, 'wb') as f:
        pickle.dump([test_loss_collect, test_acc_collect], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    if len(traning_times) > 0:
        print(f'\n Avg Training Time: {np.median(np.array(traning_times))}')
        file_path = './results/times.csv'
        os.makedirs('./results', exist_ok=True)
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(traning_times)
    else:
        print('\n Avg Training Time: N/A')

    # plotting (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    plt.figure()
    plt.title('FedASMU-FA(Server): Test Loss vs Communication rounds')
    plt.plot(range(len(test_loss_collect)), test_loss_collect)
    plt.ylabel('Test loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(
        './save/fedasmu_fa_server_{}_{}_loss.png'.format(args.dataset, args.model))

    plt.figure()
    plt.title('FedASMU-FA(Server): Test Acc vs Communication rounds')
    plt.plot(range(len(test_acc_collect)), test_acc_collect)
    plt.ylabel('Test Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(
        './save/fedasmu_fa_server_{}_{}_acc.png'.format(args.dataset, args.model))
