#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from math import ceil

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--byzantines', type=int, default=0,
                        help="number of Byzantine users who submit zero weights: Z")

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--num_users', type=int, default=21,
                        help="number of users (nodes): K")
    parser.add_argument('--frac', type=float, default=0.1,  # 10%
                        help='the fraction of clients: C')

    # fedawe arguments
    parser.add_argument('--awe_global_lr', type=float, default=1.0,
                        help='FedAWE Global LR')

    # fedaam arguments
    parser.add_argument('--aam-delta', type=float, default=0.2,
                        help='EWMA delta for e_t (Eq.(8))')
    parser.add_argument('--aam-sigma', type=float, default=0.1,
                        help='sigma for Rule-2 (Eqs.(13),(14))')
    parser.add_argument('--aam-zeta', type=float, default=0.0,
                        help='Residual zeta in Algorithm 1')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Local momentum beta (Eq.(10))')
    parser.add_argument('--lambda-scale', type=float, default=1.0,
                        help='Gradient scale lambda (Eq.(10))')

    # fedasync arguments
    parser.add_argument('--stale', type=int, default=4,
                        help='max staleness (default: 4)')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='mixing hyperparameter (default: 0.6)')

    # BRAIN arguments
    parser.add_argument('--diff', type=float, default=0.55,
                        help='the franction related to quorum (default: 0.55 => 21*0.55=11 nodes)')
    parser.add_argument('--window', type=int, default=4,
                        help='window size for moving averaging (default: 4) (>= 2)')
    parser.add_argument('--threshold', type=float, default=0.125,
                        help='accuracy threshold to ignore (default: 0.125)')
    parser.add_argument('--score_byzantines', type=int, default=0,
                        help="number of Byzantine users who submit random score: SZ")

    # FRAIN arguments
    parser.add_argument('--drift', type=int, default=0,
                        help="number of users who are drifted from global model")
    parser.add_argument('--fast_window', type=int, default=2,
                        help='window size for fast sync (default: 2) (>= 2)')
    parser.add_argument('--fast_threshold', type=float, default=0.125,
                        help='fast sync threshold to ignore (default: 0.125)')
    parser.add_argument('--interpol', type=str, default='slerp',
                        help='interpolation method (lerp or slerp)')
    parser.add_argument('--adaptive', type=str, default='constant',
                        help='adaptive mixing method (constant, poly or hinge)')
    parser.add_argument('--adaptive_a', type=float, default=0.0,
                        help='constant a in adaptive mixing method (default: 0.0)')
    parser.add_argument('--adaptive_b', type=float, default=0.0,
                        help='constant b in adaptive mixing method (default: 0.0)')
    parser.add_argument('--adaptive_c', type=float, default=4.0,
                        help='constant c in adaptive mixing method (default: 4.0)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    # parser.add_argument('--norm', type=str, default='batch_norm',
    #                     help="batch_norm, layer_norm, or None")
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--lr', type=float, default=11.5,  # 9.0 for single SGD
                        help='learning rate (default: 11.5)')

    # other arguments
    # parser.add_argument('--gpu', type=int, default=None, help="To use cuda, set \
    # to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--epochs', type=int, default=200,  # Rounds
                        help="number of rounds of training")
    parser.add_argument('--local_ep', type=float, default=9.9,  # 10
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=1024,
                        help="local batch size: B")
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='SGD momentum (default: 0.9)')
    # parser.add_argument('--optimizer', type=str, default='sgd', help="type \
    #                     of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID (and unequal).')
    # parser.add_argument('--unequal', type=int, default=0,
    #                     help='whether to use unequal data splits for  \
    #                     non-i.i.d setting (use 0 for equal splits)')
    # parser.add_argument('--stopping_rounds', type=int, default=10,
    #                     help='rounds of early stopping')
    parser.add_argument("--min_samples", type=int, default=ceil(1024/0.9),
                        help="Pareto Split min_samples")
    parser.add_argument("--min_per_label", type=int, default=128,
                        help="Pareto Split min_per_label")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    # parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args
