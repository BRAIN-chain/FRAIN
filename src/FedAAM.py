#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
import csv
from collections import OrderedDict
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from impls.options import args_parser
from impls.update import LocalUpdateFedAAM, ByzantineLocalUpdateFedAAM, test_inference
from impls.utils import get_dataset, exp_details
from impls.utils import compose_weight as compose_weight_lerp

from impls.cache import ItemCache

from airbench.model import make_net
from airbench.hyperparameters import hyp


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
    traning_times = []

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    # FedAAM Hyperparams
    aam_delta = float(getattr(args, 'aam_delta', 0.2))
    aam_sigma = float(getattr(args, 'aam_sigma', 0.1))
    aam_zeta = float(getattr(args, 'aam_zeta', 0.0))
    beta_mom = float(getattr(args, 'beta', 0.9))
    lam_scale = float(getattr(args, 'lambda_scale', 1.0))

    num_byzantines = args.byzantines

    # load dataset and user groups
    os.makedirs('./save', exist_ok=True)
    os.makedirs('./save/objects', exist_ok=True)
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if (args.model == 'cnn') and (args.dataset == 'cifar'):
        pass
    else:
        exit('Error: unrecognized model')
    # Make Model
    widths = hyp['net']['widths']
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    scaling_factor = hyp['net']['scaling_factor']
    global_model = make_net(widths, batchnorm_momentum, scaling_factor)
    global_model.train()
    print(global_model)

    # copy weights
    # global_weights = global_model.state_dict()
    global_weights = copy.deepcopy(global_model.state_dict())
    global_momentum = zeros_like_state_dict(global_weights)  # FedAAM

    test_loss_collect, test_acc_collect = [], []

    # Cache
    cache = ItemCache(min_counter=0, max_counter=args.stale)

    # FedAAM: e_t
    e_pred = max(1, int(args.frac * args.num_users))
    last_received_count = e_pred

    # FedAAM: (w_{t-1}, m_{t-1})
    prev_round_weights = copy.deepcopy(global_weights)
    prev_round_momentum = copy.deepcopy(global_momentum)

    for epoch in tqdm(range(args.epochs + args.stale)):
        if (len(cache.cache) == 0) and (epoch >= args.epochs):
            break

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
                        hyps=hyp,
                        dataset=train_dataset,
                        idxs=user_groups[idx - num_byzantines],
                        logger=logger
                    )
                    traning_start = time.time()
                else:
                    local_model = ByzantineLocalUpdateFedAAM(
                        args=args,
                        hyps=None,
                        dataset=train_dataset,
                        idxs=[],
                        logger=logger
                    )

                # w, loss = local_model.update_weights(
                #     model=copy.deepcopy(global_model),
                #     epochs=args.local_ep,
                #     global_round=epoch
                # )

                # FedAAM w/ momentum
                w_local, loss_local, m_local = local_model.update_weights(
                    model=copy.deepcopy(global_model),
                    epochs=args.local_ep,
                    global_round=epoch,
                    # TODO
                    # server_momentum=copy.deepcopy(global_momentum),
                    # beta=beta_mom,
                    # lambda_scale=lam_scale
                )

                if idx >= args.byzantines:
                    traning_times.append(time.time() - traning_start)

                # cache.add_item_with_random_counter(copy.deepcopy(w))
                cache.add_item_with_random_counter({'w': copy.deepcopy(w_local),
                                                    'm': copy.deepcopy(m_local)})

        # use counter for staleness
        # - return list of the original counters (counter == staleness)
        # - e.g.) 0 ~ 4
        # local_weights, local_stales = cache.update_counters(
        #     return_staleness=True
        # )
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
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_collect.append(test_acc)
        test_loss_collect.append(test_loss)
        # print(
        #     f'\nResults after {epoch+1}/{args.epochs+1} global rounds of training:')
        # print("Test Accuracy: {:.2f}%".format(100*test_acc))
        # print(f'Test Loss    : {format(test_loss)}')

        prev_round_weights = copy.deepcopy(round_start_weights)
        prev_round_momentum = copy.deepcopy(round_start_momentum)

    # Saving the objects test_loss_collect and test_acc_collect:
    file_name = './save/objects/fedaam_{}_{}_{}_N{}_C{}_iid{}_E{}_B{}_Z{}_S{}_delta{}_sigma{}_zeta{}_beta{}_lambda{}_{}.pkl'.\
        format(
            args.dataset, args.model, args.epochs, args.num_users, args.frac, args.iid,
            args.local_ep, args.local_bs, args.byzantines, args.stale,
            aam_delta, aam_sigma, aam_zeta, beta_mom, lam_scale,
            time.time()
        )

    with open(file_name, 'wb') as f:
        pickle.dump([test_loss_collect, test_acc_collect], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    print(f'\n Avg Training Time: {np.median(np.array(traning_times))}')
    file_path = './results/times.csv'
    os.makedirs('./results', exist_ok=True)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(traning_times)

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(test_loss_collect)), test_loss_collect, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(
        './save/fedaam_{}_{}_{}_N{}_C{}_iid{}_E{}_B{}_Z{}_S{}_delta{}_sigma{}_zeta{}_beta{}_lambda{}_loss.png'.
        format(
            args.dataset, args.model, args.epochs, args.num_users, args.frac, args.iid,
            args.local_ep, args.local_bs, args.byzantines, args.stale,
            aam_delta, aam_sigma, aam_zeta, beta_mom, lam_scale
        )
    )

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(test_acc_collect)), test_acc_collect, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(
        './save/fedaam_{}_{}_{}_N{}_C{}_iid{}_E{}_B{}_Z{}_S{}_delta{}_sigma{}_zeta{}_beta{}_lambda{}_acc.png'.
        format(
            args.dataset, args.model, args.epochs, args.num_users, args.frac, args.iid,
            args.local_ep, args.local_bs, args.byzantines, args.stale,
            aam_delta, aam_sigma, aam_zeta, beta_mom, lam_scale
        )
    )
