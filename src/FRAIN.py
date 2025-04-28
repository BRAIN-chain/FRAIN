#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import random
import pickle
import numpy as np
import statistics
import csv
from tqdm import tqdm

from tensorboardX import SummaryWriter

from impls.options import args_parser
from impls.update import LocalUpdate, ByzantineLocalUpdate, test_inference
from impls.utils import get_dataset, compose_weight_slerp, weighted_average_weights, exp_details
from impls.utils import compose_weight as compose_weight_lerp

from impls.cache import ItemCache
from impls.moving_average import MovingAverage

from airbench.model import make_net
from airbench.hyperparameters import hyp


if __name__ == '__main__':
    start_time = time.time()
    traning_times = []

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    num_byzantines = (args.byzantines if args.byzantines <
                      args.score_byzantines else args.score_byzantines)

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

    # Training
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    test_loss_collect, test_acc_collect = [], []

    # Cache
    cache = ItemCache(min_counter=0, max_counter=args.stale)

    # Moving Average
    wma = MovingAverage(args.window)

    # drift
    drifted_model = make_net(widths, batchnorm_momentum, scaling_factor)
    drifted_model.load_state_dict(global_weights)
    drifted_model.train()

    local_models = []
    scores = []

    for epoch in tqdm(range(args.epochs + args.stale)):
        if (len(cache.cache) == 0) and (epoch >= args.epochs):
            break

        local_weights = []
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        if (epoch < args.epochs):
            global_model.train()
            drifted_model.train()

            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(
                range(args.num_users), m, replace=False)

            for idx in idxs_users:
                if idx >= args.byzantines:
                    local_model = LocalUpdate(args=args, hyps=hyp,
                                              dataset=train_dataset, idxs=user_groups[idx -
                                                                                      num_byzantines],
                                              logger=logger)
                    traning_start = time.time()
                else:
                    local_model = ByzantineLocalUpdate(args=args, hyps=None,
                                                       dataset=train_dataset, idxs=[],
                                                       logger=logger)

                if args.drift == 0:  # no fast sync (same as BRAIN)
                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(global_model), epochs=args.local_ep, global_round=epoch)
                elif args.drift == args.num_users:
                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(drifted_model), epochs=args.local_ep, global_round=epoch)
                elif random.random() < args.drift / args.num_users:
                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(drifted_model), epochs=args.local_ep, global_round=epoch)
                else:
                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(global_model), epochs=args.local_ep, global_round=epoch)

                if idx >= args.byzantines:
                    traning_times.append(time.time() - traning_start)

                # local_weights.append(copy.deepcopy(w))
                cache.add_item_with_random_counter(copy.deepcopy(w), epoch)

        # use counter for staleness
        # - return list of the original counters (counter == staleness)
        # - e.g.) 0 ~ 4
        local_weights, local_staleness = cache.update_counters(
            return_epoch=True
        )

        # BRAIN: do evaluate, to get score, among randomly sampled nodes
        committee = list(range(args.num_users))
        if args.diff != 1.0:
            m = max(int(args.diff * args.num_users), 1)
            committee = np.random.choice(
                range(args.num_users), m, replace=False)

        local_eval_med_accs = []
        if len(local_weights) != 0:
            for local_weight in local_weights:
                local_eval_acc = []

                for idx in committee:
                    # BRAIN: `score_byzantines` submit random score
                    if idx >= args.score_byzantines:
                        local_model = LocalUpdate(args=args, hyps=hyp,
                                                  dataset=train_dataset, idxs=user_groups[idx -
                                                                                          num_byzantines],
                                                  logger=logger)
                        temp_model = copy.deepcopy(global_model)
                        temp_model.load_state_dict(local_weight)
                        temp_model.eval()
                        acc, loss = local_model.inference(model=temp_model)
                        local_eval_acc.append(acc)
                    else:
                        local_eval_acc.append(np.random.random())

                med_score = statistics.median(local_eval_acc)
                # BRAIN: reject updates using score by `threshold`
                # FRAIN: TODO: score functions
                # - use `local_staleness`
                # - [ ] * args: polynomial TH (a)
                # - [ ] * args: hinge THs (a, b, c)
                if med_score >= args.threshold:
                    # 1) constant
                    local_eval_med_accs.append(med_score)

                    # 2) poly (polynomial)

                    # 3) hinge

                else:
                    local_eval_med_accs.append(None)

        # agreed score
        if len(local_weights) != 0:
            for local_weight, score in zip(local_weights, local_eval_med_accs):
                if score is None:
                    pass
                else:
                    local_models.append(local_weight)
                    scores.append(score)

        # update global weights
        if len(local_weights) != 0:
            for local_weight, score in zip(local_weights, local_eval_med_accs):
                # BRAIN: aggregate updates using `window`-sized moving average
                if score is None:
                    pass
                else:
                    alpha = wma.next(score)

                    if (args.interpol == 'slerp'):
                        global_weights = compose_weight_slerp(
                            global_weights, local_weight, alpha)
                    elif (args.dataset == 'lerp'):  # same as BRAIN
                        global_weights = compose_weight_lerp(
                            global_weights, local_weight, alpha)
                    else:
                        exit('Error: unrecognized interpolation method')

                    global_model.load_state_dict(global_weights)

        # drift
        if len(local_models) != 0:
            # drifted_weights = weighted_average_weights(
            #     local_models[-1 * args.window:],
            #     scores[-1 * args.window:]
            # )
            pairs = list(zip(local_models, scores))
            filtered_pairs = []
            for model, score in reversed(pairs):
                if score >= args.fast_threshold:
                    filtered_pairs.append((model, score))
                    if len(filtered_pairs) == args.fast_window:
                        break
            if not filtered_pairs:  # fallback
                drifted_weights = weighted_average_weights(
                    local_models[-args.fast_window:],
                    scores[-args.fast_window:]
                )
            else:
                filtered_pairs.reverse()
                filtered_models, filtered_scores = zip(*filtered_pairs)
                drifted_weights = weighted_average_weights(
                    filtered_models,
                    filtered_scores
                )

            drifted_model.load_state_dict(drifted_weights)

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_collect.append(test_acc)
        test_loss_collect.append(test_loss)
        # print(
        #     f'\nResults after {epoch+1}/{args.epochs+1} global rounds of training:')
        # print("Test Accuracy: {:.2f}%".format(100*test_acc))
        # print(f'Test Loss    : {format(test_loss)}')

    # Saving the objects test_loss_collect and test_acc_collect:
    file_name = './save/objects/drift_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_D{}_W{}_S{}_TH{}_DR{}_{}.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.byzantines, args.score_byzantines,
               args.diff, args.window, args.stale, args.threshold, args.drift, time.time())

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
    plt.savefig('./save/drift_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_D{}_W{}_S{}_TH{}_DR{}_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.byzantines, args.score_byzantines,
                       args.diff, args.window, args.stale, args.threshold, args.drift))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(test_acc_collect)), test_acc_collect, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/drift_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_D{}_W{}_S{}_TH{}_DR{}_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.byzantines, args.score_byzantines,
                       args.diff, args.window, args.stale, args.threshold, args.drift))
