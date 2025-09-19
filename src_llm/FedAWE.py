import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import gc
import torch
from tensorboardX import SummaryWriter

from src_llm.impls.options import args_parser
from src_llm.impls.update import LocalUpdate, ByzantineLocalUpdate, test_inference
from src_llm.impls.utils import pareto_splits, uniform_splits, average_weights, compose_weight

from src_llm.llama.model import make_net


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
    logger = SummaryWriter('./logs_llm')

    args = args_parser()

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
    global_weights = global_model.state_dict()

    # --- AWE state: per-client local copy & unavailability counters ---
    client_states = [copy.deepcopy(global_weights)
                     for _ in range(args.num_users)]
    miss_counts = np.ones(args.num_users, dtype=np.int64)

    # --- AWE hyperparameter: global step size eta_g ---
    AWE_GLOBAL_LR = args.awe_global_lr

    # TRAIN

    ppl_collect = []

    # before training
    ppl, ppl_std = test_inference(args, global_model, tokenizer)
    print(f"[INFO] Wikitext PPL (wikitext2): {ppl} +- {ppl_std}")
    ppl_collect.append(ppl)

    pbar = tqdm(range(args.epochs), desc="Epochs")
    for epoch in pbar:
        # try:
        local_weights = []
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            if idx >= args.byzantines:
                local_model = LocalUpdate(
                    args=args,
                    tokenizer=tokenizer
                )
                # trainset set
                (skip_num, take_num) = user_groups[idx-args.byzantines]
                local_model.set_dataset_train(
                    skip_num, take_num,
                    seed=int(time.time() * 1000) & (2**32 - 1)
                )
            else:
                local_model = ByzantineLocalUpdate(
                    args=args, tokenizer=tokenizer
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

        # Test inference after completion of training
        ppl, ppl_std = test_inference(args, global_model, tokenizer)
        print(f"[INFO] Wikitext PPL (wikitext2): {ppl} +- {ppl_std}")
        ppl_collect.append(ppl)

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
    file_name = './save_llm/objects/fedawe_{}_C{}_iid{}_E{}_B{}_Z{}_LR{}_{}.pkl'.\
        format(args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.byzantines, args.awe_global_lr, time.time())

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
    plt.savefig('./save_llm/fedawe_{}_C{}_iid{}_E{}_B{}_Z{}_LR{}_ppl.png'.
                format(args.epochs, args.frac, args.iid,
                       args.local_ep, args.local_bs, args.byzantines, args.awe_global_lr))
