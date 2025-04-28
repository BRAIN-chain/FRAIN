#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import time
import copy
from math import ceil

import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    def print_label_distribution(user_data, dataset):
        num_labels = 10  # CIFAR10 has 10 classes
        label_names = dataset.classes

        # For each user
        for user, indices in user_data.items():
            # Initialize a count dictionary for labels
            label_counts = {label: 0 for label in label_names}

            # Count labels
            for idx in indices:
                label = dataset.targets[idx]
                label_counts[label_names[label]] += 1

            # Print distribution for the user
            print(f'User {user} Label Distribution:')
            for label, count in label_counts.items():
                print(f'{label}: {count}')
            print('')  # Print a newline for better separation

    def plot_label_distribution(user_data, dataset):
        num_labels = 10  # CIFAR10 has 10 classes
        num_users = len(user_data)
        label_names = dataset.classes
        user_names = [rf'${i}$' for i in range(num_users)]

        # Initialize the matrix to hold label counts for each user
        label_counts_matrix = np.zeros((num_labels, num_users), dtype=int)

        for user, indices in user_data.items():
            for idx in indices:
                label = dataset.targets[idx]
                label_counts_matrix[label, user] += 1

        # Create the heatmap
        vmax = ceil(float(np.max(label_counts_matrix))/100)*100
        plt.figure(figsize=(10.5, 4.5))
        ax = sns.heatmap(label_counts_matrix, annot=True, fmt="d",
                         cmap="Greys",
                         vmin=0, vmax=vmax,
                         xticklabels=user_names, yticklabels=label_names)

        # ax.set_title('Label Distribution Across Users')
        # ax.set_xlabel('Node')
        # ax.set_ylabel('Label')
        plt.yticks(rotation=45)

        cbar = ax.collections[0].colorbar
        cbar.outline.set_linewidth(1)

        plt.tight_layout()
        plt.savefig(
            f'./save/distribution_{args.dataset}_iid{args.iid}_{int(time.time())}.png', bbox_inches='tight', dpi=300)

    if args.dataset == 'cifar':
        data_dir = './data/cifar10'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            user_groups = cifar_iid(
                train_dataset, args.num_users - (args.byzantines if args.byzantines < args.score_byzantines else args.score_byzantines))
        else:
            user_groups = cifar_noniid(
                train_dataset, args.num_users - (args.byzantines if args.byzantines < args.score_byzantines else args.score_byzantines), alpha=3.0)
    else:
        raise NotImplementedError()

    # Visualization
    # print([len(user_groups[k]) for k in user_groups.keys()])
    plot_label_distribution(user_groups, train_dataset)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def weighted_average_weights(w, a):
    """
    Returns the weighted average of the weights.
    """
    denom = sum(a)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], a[0])
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * a[i]
        w_avg[key] = torch.div(w_avg[key], denom)
    return w_avg


def compose_weight(w0, w1, a=0.6):
    """
    Returns the average of the weights.
    """
    w_t = copy.deepcopy(w0)
    for key in w_t.keys():
        w_t[key] = (1.0-a) * w0[key] + a * w1[key]
    return w_t


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Dataset   : {args.dataset}/{args.num_classes}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
