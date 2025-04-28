#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from math import ceil

import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users, alpha=3.0, min_samples=ceil(1024/0.9), min_per_label=100):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :param alpha: shape parameter of the Pareto distribution. Default is 3.0, 
                a common choice to simulate imbalance. Lower values lead to 
                higher imbalance among users.
    :param min_samples: minimum number of samples each user should receive
    :param min_per_label: minimum number of samples per (node, label)
    :return:
    """
    num_labels = 10  # CIFAR-10 has 10 labels

    # Ensure the dataset can support the min_samples for each user
    if (num_users * min_samples > len(dataset)) and (num_users * num_labels > len(dataset)):
        raise ValueError(
            "The total min_samples for all users exceed the total number of items in the dataset.")

    # Group data by labels
    label_to_indices = {
        i: np.where(np.array(dataset.targets) == i)[0] for i in range(num_labels)}

    # Initialize user data distribution
    user_data = {i: [] for i in range(num_users)}

    for label, indices in label_to_indices.items():
        np.random.shuffle(indices)
        # Split label indices among users based on Pareto distribution
        samples_pareto = np.random.pareto(alpha, num_users)
        samples_pareto_normalized = samples_pareto / \
            sum(samples_pareto) * (len(indices) - min_per_label * num_users)
        samples_per_user = [int(np.round(num))
                            for num in samples_pareto_normalized]

        # Adjust last to match exactly
        samples_per_user[-1] = len(indices) - sum(samples_per_user[:-1])
        samples_per_user = np.array(samples_per_user) + min_per_label

        # Distribute indices among users
        start = 0
        for user, num_samples in enumerate(samples_per_user):
            user_data[user].extend(indices[start:start + num_samples])
            start += num_samples

    # Enforcing minimum samples by redistributing excess samples from users who have more than minimum
    redistribute_indices = []
    for user, indices in user_data.items():
        if len(indices) < min_samples:
            needed = min_samples - len(indices)
            for donor_user, donor_indices in user_data.items():
                if len(donor_indices) > min_samples + needed:
                    transfer_indices = donor_indices[-needed:]
                    donor_indices = donor_indices[:-needed]
                    user_data[donor_user] = donor_indices
                    redistribute_indices.extend(transfer_indices)
                    break

    # Distribute any collected indices for redistribution
    np.random.shuffle(redistribute_indices)
    for user, indices in user_data.items():
        if len(indices) < min_samples:
            needed = min_samples - len(indices)
            transfer_indices = redistribute_indices[:needed]
            redistribute_indices = redistribute_indices[needed:]
            user_data[user].extend(transfer_indices)

    # Convert lists to sets for consistency with the previous function's output
    user_data = {user: set(indices) for user, indices in user_data.items()}

    return user_data


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
