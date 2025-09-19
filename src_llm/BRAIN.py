import os
import copy
import time
import pickle
import numpy as np
import statistics

import gc
import torch
from tensorboardX import SummaryWriter

from src_llm.impls.options import args_parser
from src_llm.impls.update import LocalUpdate, ByzantineLocalUpdate, test_inference
from src_llm.impls.cache import ItemCache
from src_llm.impls.moving_average import MovingAverage
from src_llm.impls.utils import pareto_splits, uniform_splits, average_weights, compose_weight, exp_decay

from src_llm.llama.model import make_net


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

    # TRAIN

    ppl_collect = []

    # before training
    ppl, ppl_std = test_inference(args, global_model, tokenizer)
    print(f"[INFO] Wikitext PPL (wikitext2): {ppl} +- {ppl_std}")
    ppl_collect.append(ppl)

    # Cache
    cache = ItemCache(min_counter=0, max_counter=args.stale)

    # Moving Average
    wma = MovingAverage(args.window)

    for epoch in range(args.epochs + args.stale):
        print("[INFO]", epoch)

        if (len(cache.cache) == 0) and (epoch >= args.epochs):
            break

        try:
            local_weights = []
            # print(f'\n | Global Training Round : {epoch+1} |\n')

            if (epoch < args.epochs):
                global_model.train()
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(
                    range(args.num_users), m, replace=False)

                for idx in idxs_users:
                    if idx >= args.byzantines:
                        local_model = LocalUpdate(
                            args=args, tokenizer=tokenizer)
                        # trainset set
                        (skip_num, take_num) = user_groups[idx-args.byzantines]
                        local_model.set_dataset_train(
                            skip_num, take_num, seed=int(time.time() * 1000) & (2**32 - 1))
                    else:
                        local_model = ByzantineLocalUpdate(
                            args=args, tokenizer=tokenizer)

                    w, avg_train_loss = local_model.update_weights(
                        model=copy.deepcopy(global_model),
                        epochs=args.local_ep,
                        global_round=epoch
                    )

                    # local_weights.append(copy.deepcopy(w))
                    cache.add_item_with_random_counter(copy.deepcopy(w))

            local_weights = cache.update_counters()

            # BRAIN: do evaluate, to get score, among randomly sampled nodes
            committee = list(range(args.num_users))
            if args.diff != 1.0:
                m = max(int(args.diff * args.num_users), 1)
                committee = np.random.choice(
                    range(args.num_users), m, replace=False)

            local_eval_val_loss = []
            if len(local_weights) != 0:
                for local_weight in local_weights:
                    local_eval_loss = []

                    for idx in committee:
                        # BRAIN: `score_byzantines` submit random score
                        if idx >= args.score_byzantines:
                            local_model = LocalUpdate(
                                args=args, tokenizer=tokenizer)
                            temp_model = copy.deepcopy(global_model)
                            temp_model.load_state_dict(local_weight)
                            temp_model.eval()
                            val_loss = local_model.inference(model=temp_model)
                            # scale loss (use exp, e^-ax)
                            local_eval_loss.append(
                                exp_decay(val_loss, alpha=0.1))  # TODO: alpha=0.05?
                        else:
                            # scale 0.0~1.0
                            local_eval_loss.append(np.random.random())

                    med_score = statistics.median(local_eval_loss)
                    # BRAIN: reject updates using score by `threshold`
                    if med_score >= args.threshold:
                        local_eval_val_loss.append(med_score)
                    else:
                        local_eval_val_loss.append(None)

            # update global weights
            if len(local_weights) != 0:
                for local_weight, score in zip(local_weights, local_eval_val_loss):
                    # BRAIN: aggregate updates using `window`-sized moving average
                    if score is None:
                        pass
                    else:
                        alpha = wma.next(score)
                        global_weights = compose_weight(
                            global_weights, local_weight, alpha)
                        global_model.load_state_dict(global_weights)

            # Test inference after completion of training
            ppl, ppl_std = test_inference(args, global_model, tokenizer)
            print(f"[INFO] Wikitext PPL (wikitext2): {ppl} +- {ppl_std}")
            ppl_collect.append(ppl)

            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:  # TODO: print error msg
            # if 'out of memory' in str(e):
            print(
                f"[WARNING] CUDA OOM at epoch {epoch}, stopping training loop.")
            torch.cuda.empty_cache()
            args.epochs = epoch
            break  # or continue
            # else:
            # raise

    # Saving the objects:
    file_name = './save_llm/objects/brain_{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_D{}_W{}_S{}_TH{}_{}.pkl'.\
        format(args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.byzantines, args.score_byzantines,
               args.diff, args.window, args.stale, args.threshold, time.time())

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
    plt.savefig('./save_llm/brain_{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_D{}_W{}_S{}_TH{}_ppl.png'.
                format(args.epochs, args.frac, args.iid,
                       args.local_ep, args.local_bs, args.byzantines, args.score_byzantines,
                       args.diff, args.window, args.stale, args.threshold))
