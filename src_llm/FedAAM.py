import os
import copy
import time
import pickle
import numpy as np
import statistics
from collections import OrderedDict
from tqdm import tqdm

import gc
import torch
from tensorboardX import SummaryWriter

from src_llm.impls.options import args_parser
from src_llm.impls.update import LocalUpdateFedAAM, ByzantineLocalUpdateFedAAM, test_inference
from src_llm.impls.utils import pareto_splits, uniform_splits
from src_llm.impls.utils import compose_weight as compose_weight_lerp

from src_llm.impls.cache import ItemCache

from src_llm.llama.model import make_net


# ---------------- FedAAM helpers ----------------
def zeros_like_state_dict(sd):
    return OrderedDict((k, torch.zeros_like(v)) for k, v in sd.items())


def lerp_state_dict(a, b, alpha: float):
    if a is None or b is None:
        return None
    out = OrderedDict()
    a1 = float(1.0 - alpha)
    a2 = float(alpha)
    for k in a.keys():
        out[k] = a1 * a[k] + a2 * b[k]
    return out


def fedaam_predict_e(e_prev: float, last_count: int, delta: float) -> float:
    return (1.0 - float(delta)) * float(e_prev) + float(delta) * float(last_count)


def fedaam_make_alphas(num_updates: int, e_pred: float, zeta: float):
    e = max(1, int(round(e_pred)))
    g = (1.0 - float(zeta)) / float(e)
    alphas = [0.0] * num_updates
    prod_after = 1.0
    for r in reversed(range(num_updates)):
        if (r + 1) <= e:
            a = g / max(prod_after, 1e-12)
        else:
            a = (1.0 - float(zeta)) / float(r + 1)
        a = min(max(a, 0.0), 1.0)
        alphas[r] = a
        prod_after *= (1.0 - a)
    return alphas


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs_llm')

    args = args_parser()

    # FedAAM Hyperparams
    aam_delta = float(getattr(args, 'aam_delta', 0.2))
    aam_sigma = float(getattr(args, 'aam_sigma', 0.1))
    aam_zeta = float(getattr(args, 'aam_zeta', 0.0))
    beta_mom = float(getattr(args, 'beta', 0.9))
    lam_scale = float(getattr(args, 'lambda_scale', 1.0))

    # load dataset and user groups
    os.makedirs('./save_llm', exist_ok=True)
    os.makedirs('./save_llm/objects', exist_ok=True)

    if args.iid:
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
    # for user_idx, (skip, take) in enumerate(user_groups):
    #     pass

    # Make Model
    model_name = "HuggingFaceTB/SmolLM2-135M"
    # model_name = "meta-llama/Llama-3.2-1B"
    global_model, tokenizer = make_net(
        model_name=model_name,
    )
    global_model.train()
    print(global_model)

    # copy weights
    # global_weights = global_model.state_dict()
    global_weights = copy.deepcopy(global_model.state_dict())
    global_momentum = zeros_like_state_dict(global_weights)  # FedAAM

    # TRAIN

    ppl_collect = []

    # before training
    ppl, ppl_std = test_inference(args, global_model, tokenizer)
    print(f"[INFO] Wikitext PPL (wikitext2): {ppl} +- {ppl_std}")
    ppl_collect.append(ppl)

    # Cache
    cache = ItemCache(min_counter=0, max_counter=args.stale)

    # FedAAM: e_t
    e_pred = max(1, int(args.frac * args.num_users))
    last_received_count = e_pred

    # FedAAM: (w_{t-1}, m_{t-1})
    prev_round_weights = copy.deepcopy(global_weights)
    prev_round_momentum = copy.deepcopy(global_momentum)

    # pbar = tqdm(range(args.epochs + args.stale), desc="Epochs")
    # for epoch in pbar:
    for epoch in range(args.epochs + args.stale):
        print("[INFO]", epoch)

        if (len(cache.cache) == 0) and (epoch >= args.epochs):
            break

        # try:
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        if (epoch < args.epochs):
            global_model.train()

            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(
                range(args.num_users), m, replace=False)

            for idx in idxs_users:
                if idx >= args.byzantines:
                    local_model = LocalUpdateFedAAM(
                        args=args,
                        tokenizer=tokenizer
                    )
                    # trainset set
                    (skip_num, take_num) = user_groups[idx-args.byzantines]
                    local_model.set_dataset_train(
                        skip_num, take_num, seed=int(time.time() * 1000) & (2**32 - 1))
                else:
                    local_model = ByzantineLocalUpdateFedAAM(
                        args=args,
                        tokenizer=tokenizer
                    )

                # w, avg_train_loss = local_model.update_weights(
                #     model=copy.deepcopy(global_model),
                #     epochs=args.local_ep,
                #     global_round=epoch
                # )

                # FedAAM w/ momentum
                w_local, avg_train_loss_local, m_local = local_model.update_weights(
                    model=copy.deepcopy(global_model),
                    epochs=args.local_ep,
                    global_round=epoch,
                    # TODO
                    # server_momentum=copy.deepcopy(global_momentum),
                    # beta=beta_mom,
                    # lambda_scale=lam_scale
                )

                # cache.add_item_with_random_counter(copy.deepcopy(w))
                cache.add_item_with_random_counter({'w': copy.deepcopy(w_local),
                                                    'm': copy.deepcopy(m_local)})

        # local_weights = cache.update_counters()
        payloads, stales = cache.update_counters(return_staleness=True)
        # FedAAM: (stale==0: moderate(v=t), stale==1: fast(v=t-1))
        # only use recent two
        accepted = [(p, s) for p, s in zip(payloads, stales) if s <= 1]
        R = len(accepted)

        # FedAAM
        round_start_weights = copy.deepcopy(global_weights)
        round_start_momentum = copy.deepcopy(global_momentum)

        if R > 0:
            alphas = fedaam_make_alphas(R, e_pred, aam_zeta)

            prefix_prod = 1.0

            for r_idx, (payload, stale) in enumerate(accepted):
                a = float(alphas[r_idx])
                a_eff = a / float(stale + 1)
                a_eff = min(max(a_eff, 0.0), 1.0)

                prefix_prod *= (1.0 - a_eff)

                if stale == 1:
                    global_weights = compose_weight_lerp(
                        global_weights,  payload['w'], a_eff)
                    global_momentum = lerp_state_dict(
                        global_momentum, payload['m'], a_eff)

                else:  # stale == 0
                    sigma = float(aam_sigma)
                    scale = prefix_prod * sigma

                    # w <- w + Pσ (w_{t-1} - w_t)
                    w_out = OrderedDict()
                    for k in global_weights.keys():
                        w_out[k] = global_weights[k] + scale * \
                            (prev_round_weights[k] - round_start_weights[k])
                    global_weights = w_out

                    # m <- m + Pσ (m_{t-1} - m_t)
                    m_out = OrderedDict()
                    if global_momentum is not None:
                        for k in global_momentum.keys():
                            m_out[k] = global_momentum[k] + scale * \
                                (prev_round_momentum[k] -
                                 round_start_momentum[k])
                        global_momentum = m_out

            last_received_count = R
        else:
            last_received_count = 0

        # FedAAM
        e_pred = fedaam_predict_e(
            e_pred, last_received_count, aam_delta)

        global_model.load_state_dict(global_weights)

        # Test inference after completion of training
        ppl, ppl_std = test_inference(args, global_model, tokenizer)
        print(f"[INFO] Wikitext PPL (wikitext2): {ppl} +- {ppl_std}")
        ppl_collect.append(ppl)

        prev_round_weights = copy.deepcopy(round_start_weights)
        prev_round_momentum = copy.deepcopy(round_start_momentum)

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

    # Saving the objects:
    file_name = './save_llm/objects/fedaam_{}_C{}_iid{}_E{}_B{}_Z{}_S{}_delta{}_sigma{}_zeta{}_beta{}_lambda{}_{}.pkl'.\
        format(
            args.epochs, args.frac, args.iid,
            args.local_ep, args.local_bs, args.byzantines, args.stale,
            aam_delta, aam_sigma, aam_zeta, beta_mom, lam_scale,
            time.time()
        )

    with open(file_name, 'wb') as f:
        pickle.dump([ppl_collect], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Test PPL
    plt.figure()
    plt.title('PPL vs Communication rounds (wikitext)')
    plt.plot(range(len(ppl_collect)), ppl_collect, color='k')
    plt.ylabel('Test PPL')
    plt.xlabel('Communication Rounds')
    plt.savefig(
        './save_llm/fedaam_{}_C{}_iid{}_E{}_B{}_Z{}_S{}_delta{}_sigma{}_zeta{}_beta{}_lambda{}_ppl.png'.
        format(
            args.epochs, args.frac, args.iid,
            args.local_ep, args.local_bs, args.byzantines, args.stale,
            aam_delta, aam_sigma, aam_zeta, beta_mom, lam_scale
        )
    )
