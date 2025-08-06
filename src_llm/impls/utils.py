import copy
# from math import ceil

import torch

import numpy as np


def exp_decay(loss, alpha=0.1):
    loss = np.asarray(loss, dtype=float)
    return np.exp(-alpha * loss)


def uniform_splits(dataset_size: int,
                   num_users: int):
    """
    Generate (skip, take) splits for a sorted dataset of length `dataset_size`
    among `num_users`, dividing the data as equally as possible.

    Each user gets either floor(N/num_users) or ceil(N/num_users) items,
    with the extra 1-item allocations going to the first few users.
    """
    if num_users <= 0:
        raise ValueError("num_users must be a positive integer")
    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative")

    base_take = dataset_size // num_users
    remainder = dataset_size % num_users

    # First `remainder` users get (base_take + 1), the rest get base_take
    takes = [base_take + (1 if i < remainder else 0) for i in range(num_users)]

    splits = []
    current_index = 0
    for t in takes:
        splits.append((current_index, t))
        current_index += t

    # Sanity check
    assert current_index == dataset_size, (
        f"Total allocated {current_index} items, "
        f"but dataset_size is {dataset_size}"
    )

    return splits


def pareto_splits(dataset_size: int,
                  num_users: int,
                  alpha: float = 3.0,
                  min_per_label: int = 128,
                  seed: int = 42):
    """
    Generate (skip, take) splits for a sorted dataset of length `dataset_size` among `num_users`,
    following a Pareto distribution with shape parameter `alpha`, and ensuring each user
    receives at least `min_per_label` items.
    """
    if seed is not None:
        np.random.seed(seed)

    samples = np.random.pareto(alpha, num_users)

    # Compute how many items are reserved by the guaranteed minimum per user
    reserved = min_per_label * num_users
    if reserved > dataset_size:
        raise ValueError(
            "dataset_size is smaller than the total guaranteed minimum.")
    remaining = dataset_size - reserved

    # Compute raw allocation proportions
    proportions = samples / samples.sum()
    raw_alloc = proportions * remaining

    # Apply floor to get integer allocations
    floor_alloc = np.floor(raw_alloc).astype(int)
    alloc = floor_alloc.copy()

    # Distribute any leftover items by largest fractional remainders
    leftover = remaining - floor_alloc.sum()
    if leftover > 0:
        remainders = raw_alloc - floor_alloc
        # sort users by descending remainder
        idx_desc = np.argsort(-remainders)
        for idx in idx_desc[:leftover]:
            alloc[idx] += 1

    # Add the guaranteed minimum to each allocation
    takes = alloc + min_per_label  # total items per user

    # Build (skip, take) pairs
    splits = []
    current_index = 0
    for t in takes:
        splits.append((current_index, int(t)))
        current_index += int(t)

    # Verify we used the entire dataset
    assert current_index == dataset_size, f"Total allocated {current_index} vs dataset_size {dataset_size}"
    return splits


def _to_cpu(w):
    return {k: v.to("cpu") for k, v in w.items()}


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w = [_to_cpu(wi) for wi in w]

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        # if "lora" in key:
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
        # else:
        #     pass
    return w_avg


def weighted_average_weights(w, a):
    """
    Returns the weighted average of the weights.
    """
    w = [_to_cpu(wi) for wi in w]

    denom = max(sum(a), 1e-8)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        # if "lora" in key:
        w_avg[key] = torch.mul(w_avg[key], a[0])
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * a[i]
        w_avg[key] = torch.div(w_avg[key], denom)
    return w_avg


def compose_weight(w0, w1, a=0.6):  # LERP
    """
    Returns the average of the weights.
    """
    w0 = _to_cpu(w0)
    w1 = _to_cpu(w1)

    w_t = copy.deepcopy(w0)
    for key in w_t.keys():
        # if "lora" in key:
        w_t[key] = (1.0-a) * w0[key] + a * w1[key]
    return w_t


def compose_weight_slerp(w0, w1, a=0.6, DOT_THRESHOLD=0.9995, eps=1e-8):
    """
    Returns the SLERP-based merge of w0 and w1.

    References:
    - https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    - https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/slerp.py#L100
    """
    w0 = _to_cpu(w0)
    w1 = _to_cpu(w1)

    w_t = copy.deepcopy(w0)

    def normalize(v, eps):
        v = v.float()

        norm_v = torch.norm(v, p=2)
        if norm_v > eps:
            v = v / norm_v
        return v

    def slerp_param(v0, v1, alpha):
        v0_copy = v0.clone()
        v1_copy = v1.clone()

        v0_norm = normalize(v0_copy, eps)
        v1_norm = normalize(v1_copy, eps)

        dot = torch.sum(v0_norm * v1_norm)

        # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
        if torch.abs(dot) > DOT_THRESHOLD:
            return (1.0 - alpha) * v0_copy + alpha * v1_copy

        # SLERP
        theta_0 = torch.acos(dot)
        theta_t = alpha * theta_0
        sin_theta_0 = torch.sin(theta_0)
        sin_theta_t = torch.sin(theta_t)

        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        return s0 * v0_copy + s1 * v1_copy

    for key in w_t.keys():
        # if "lora" in key:
        w_t[key] = slerp_param(w0[key], w1[key], a)
    return w_t


if __name__ == "__main__":
    N = 36_718
    U = 21 - 5
    splits = pareto_splits(N, U, alpha=3.0, min_per_label=512)
    # splits = uniform_splits(N, U)

    for user_idx, (skip, take) in enumerate(splits):
        print(f"user {user_idx:2d}: skip={skip:5d}, take={take:5d}")
