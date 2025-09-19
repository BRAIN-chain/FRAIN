#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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

    # learning-related arguments
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')
    # parser.add_argument("--warmup_steps", type=int, default=100,
    #                     help="Number of warmup steps")
    #
    parser.add_argument('--epochs', type=int, default=25,
                        help="number of rounds of training")
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument("--eval_bs", type=int, default=16,
                        help="Batch size for evaluation")
    #
    parser.add_argument("--num_train_steps", type=int, default=4096,  # wikitext2: 36718
                        help="Number of training samples to stream")
    parser.add_argument("--num_val_steps", type=int, default=128,  # TODO: 3760
                        help="Number of validation samples to stream")
    parser.add_argument("--num_eval_steps", type=int, default=1024,  # TODO: 4358
                        help="Amount of data to sample during evaluation")

    # other arguments
    # parser.add_argument('--gpu', type=int, default=None, help="To use cuda, set \
    # to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID (and unequal).')
    parser.add_argument("--min_per_label", type=int, default=128,
                        help="Pareto Split min_per_label")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    # parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args
