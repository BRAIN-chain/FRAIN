# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FedAvg.py --iid=0 --epochs=200 --byzantines=0 --frac=0.1 --verbose=0 --awe_global_lr=1.0

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from impls.options import args_parser
from impls.update import LocalUpdate, ByzantineLocalUpdate, test_inference
from impls.utils import get_dataset, average_weights, exp_details

from airbench.model import make_net
from airbench.hyperparameters import hyp


# --- AWE helper: interpolate x_pre -> x_post with echo & global step ---
def awe_interpolate(pre_state, post_state, echo_factor=1.0, eta_g=1.0):
    # x_dagger = pre + (eta_g * echo) * (post - pre)
    out = {}
    alpha = float(eta_g) * float(echo_factor)
    for k, v_pre in pre_state.items():
        v_post = post_state[k]
        v_pre_fp = v_pre.to(dtype=v_post.dtype, device=v_post.device)
        out[k] = v_pre_fp + (v_post - v_pre_fp) * alpha
    return out


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

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
    global_weights = global_model.state_dict()

    # --- AWE state: per-client local copy & unavailability counters ---
    client_states = [copy.deepcopy(global_weights)
                     for _ in range(args.num_users)]
    miss_counts = np.ones(args.num_users, dtype=np.int64)

    # --- AWE hyperparameter: global step size eta_g ---
    AWE_GLOBAL_LR = args.awe_global_lr

    # Training
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    test_loss_collect, test_acc_collect = [], []

    pbar = tqdm(range(args.epochs), desc="Epochs")
    for epoch in pbar:
        local_weights = []

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            if idx >= args.byzantines:
                local_model = LocalUpdate(
                    args=args, hyps=hyp,
                    dataset=train_dataset,
                    idxs=user_groups[idx - args.byzantines],
                    logger=logger
                )
            else:
                local_model = ByzantineLocalUpdate(
                    args=args, hyps=None,
                    dataset=train_dataset, idxs=[],
                    logger=logger
                )

            # --- AWE: local stale model
            model_i = copy.deepcopy(global_model)
            model_i.load_state_dict(client_states[idx])

            w_post, loss = local_model.update_weights(
                model=model_i,
                epochs=args.local_ep,
                global_round=epoch
            )
            # print(f'Train Loss   : {format(loss)}')
            pbar.set_postfix(train_loss=f"{loss}")

            # --- AWE: client_states[idx]
            x_pre = client_states[idx]
            echo = miss_counts[idx]  # 0,1,2,...

            # --- AWE: x_dagger = x_pre + ηg * echo * (x_post - x_pre)
            x_dagger = awe_interpolate(
                x_pre, w_post, echo_factor=echo, eta_g=AWE_GLOBAL_LR)
            local_weights.append(x_dagger)

        # --- AWE: miss counts
        miss_counts += 1
        for idx in idxs_users:
            miss_counts[idx] = 1

        # --- AWE aggregator: x_{t+1} = average_i x_i^{\dagger}
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # --- AWE implicit gossiping
        for idx in idxs_users:
            client_states[idx] = copy.deepcopy(global_weights)

        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_collect.append(test_acc)
        test_loss_collect.append(test_loss)
        # print(
        #     f'\nResults after {epoch+1}/{args.epochs+1} global rounds of training:')
        # print("Test Accuracy: {:.2f}%".format(100*test_acc))
        # print(f'Test Loss    : {format(test_loss)}')

    # Saving the objects test_loss_collect and test_acc_collect:
    file_name = './save/objects/fedawe_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_LR{}_{}.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.byzantines, args.awe_global_lr,
               time.time())

    with open(file_name, 'wb') as f:
        pickle.dump([test_loss_collect, test_acc_collect], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

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
    plt.savefig('./save/fedawe_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_LR{}_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.byzantines, args.awe_global_lr))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(test_acc_collect)), test_acc_collect, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fedawe_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_LR{}_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.byzantines, args.awe_global_lr))
