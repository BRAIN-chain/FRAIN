import os
import copy
import time
import pickle
import numpy as np
import statistics
import random

import gc
import torch
from tensorboardX import SummaryWriter

from src_llm.impls.options import args_parser
from src_llm.impls.update import LocalUpdate, ByzantineLocalUpdate, test_inference
from src_llm.impls.cache import ItemCache
from src_llm.impls.moving_average import MovingAverage
from src_llm.impls.utils import pareto_splits, uniform_splits, weighted_average_weights, compose_weight_slerp, exp_decay
from src_llm.impls.utils import compose_weight as compose_weight_lerp

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
            min_per_label=args.min_per_label,
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
    global_weights = {k: v.cpu() for k, v in global_model.state_dict().items()}

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

    # drift
    drifted_model, _ = make_net(model_name=model_name)
    drifted_model.load_state_dict(global_weights)
    drifted_model.train()

    local_models = []
    scores = []

    for epoch in range(args.epochs + args.stale):
        print("[INFO] Epoch", epoch)
        global_model.to('cpu')
        drifted_model.to('cpu')

        if (len(cache.cache) == 0) and (epoch >= args.epochs):
            break

        try:
            local_weights = []
            # print(f'\n | Global Training Round : {epoch+1} |\n')

            if (epoch < args.epochs):
                global_model.train()
                drifted_model.train()

                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(
                    range(args.num_users), m, replace=False)

                for nidx, idx in enumerate(idxs_users):
                    print(f"Epoch {epoch}, {nidx+1}/{len(idxs_users)} ")

                    if idx >= args.byzantines:
                        local_model = LocalUpdate(
                            args=args, tokenizer=tokenizer, gpu=torch.cuda.current_device())
                        # trainset set
                        (skip_num, take_num) = user_groups[idx-args.byzantines]
                        local_model.set_dataset_train(
                            skip_num, take_num, seed=int(time.time() * 1000) & (2**32 - 1))
                    else:
                        local_model = ByzantineLocalUpdate(
                            args=args, tokenizer=tokenizer, gpu=torch.cuda.current_device())

                    if args.drift == 0:  # no fast sync (same as BRAIN)
                        w, avg_train_loss = local_model.update_weights(
                            model=copy.deepcopy(global_model),
                            epochs=args.local_ep,
                            global_round=epoch
                        )
                    elif args.drift == args.num_users:
                        w, avg_train_loss = local_model.update_weights(
                            model=copy.deepcopy(drifted_model),
                            epochs=args.local_ep,
                            global_round=epoch
                        )
                    elif random.random() < args.drift / args.num_users:
                        w, avg_train_loss = local_model.update_weights(
                            model=copy.deepcopy(drifted_model),
                            epochs=args.local_ep,
                            global_round=epoch
                        )
                    else:
                        w, avg_train_loss = local_model.update_weights(
                            model=copy.deepcopy(global_model),
                            epochs=args.local_ep,
                            global_round=epoch
                        )

                    # local_weights.append(copy.deepcopy(w))

                    # if next(iter(w.values())).is_cuda:
                    #     w = {k: v.cpu() for k, v in w.items()}
                    cache.add_item_with_random_counter(
                        w
                        # copy.deepcopy(w)
                    )

            # use counter for staleness
            # - return list of the original counters (counter == staleness)
            # - e.g.) 0 ~ 4
            local_weights, local_stales = cache.update_counters(
                return_staleness=True
            )

            # BRAIN: do evaluate, to get score, among randomly sampled nodes
            print(f"Epoch {epoch}, evaluate")
            committee = list(range(args.num_users))
            if args.diff != 1.0:
                m = max(int(args.diff * args.num_users), 1)
                committee = np.random.choice(
                    range(args.num_users), m, replace=False)

            local_eval_val_loss = []
            if len(local_weights) != 0:
                for local_weight, local_stale in zip(local_weights, local_stales):
                    local_eval_loss = []

                    for idx in committee:
                        # BRAIN: `score_byzantines` submit random score
                        if idx >= args.score_byzantines:
                            with torch.no_grad():
                                local_model = LocalUpdate(
                                    args=args, tokenizer=tokenizer, gpu=torch.cuda.current_device())
                                temp_model = copy.deepcopy(
                                    global_model).to('cpu')
                                temp_model.load_state_dict(local_weight)
                                temp_model = temp_model.to('cuda')
                                temp_model.eval()
                                val_loss = local_model.inference(
                                    model=temp_model)
                                temp_model.to('cpu')
                                del temp_model
                                # torch.cuda.empty_cache()  # too slow

                                # scale loss (use exp, e^-ax)
                                local_eval_loss.append(
                                    exp_decay(val_loss, alpha=0.1))  # TODO: alpha=0.05?
                        else:
                            # scale 0.0~1.0
                            local_eval_loss.append(np.random.random())

                    med_score = statistics.median(local_eval_loss)
                    # BRAIN: reject updates using score by `threshold`
                    if med_score >= args.threshold:
                        # FRAIN: score functions
                        if (args.adaptive == 'constant'):
                            pass  # med_score = med_score
                        elif (args.adaptive == 'poly'):
                            # args: polynomial (a)
                            # use `local_stale`
                            # med_score *= 1 / (1+stale)^a
                            decay_factor = 1.0 / \
                                ((1.0 + (local_stale)) ** args.adaptive_a)
                            med_score *= decay_factor
                        elif (args.adaptive == 'hinge'):
                            # args: (a, b, c)
                            # use `local_stale`
                            # med_score *= (1/(a(stale-b)+1)-t)/(1-t), t = 1/(a(c-b)+1)
                            if local_stale <= args.adaptive_b:
                                decay_factor = 1.0
                            elif local_stale >= args.adaptive_c:
                                decay_factor = 0.0
                            else:
                                t = 1.0 / (args.adaptive_a *
                                           (args.adaptive_c - args.adaptive_b) + 1.0)
                                numerator = 1.0 / \
                                    (args.adaptive_a *
                                        (local_stale - args.adaptive_b) + 1.0) - t
                                denominator = 1.0 - t
                                decay_factor = numerator / denominator
                            med_score *= decay_factor
                        else:
                            exit('Error: unrecognized adative mixing method')

                        local_eval_val_loss.append(med_score)
                    else:
                        local_eval_val_loss.append(None)

            # agreed score
            print(f"Epoch {epoch}, score")
            if len(local_weights) != 0:
                for local_weight, score in zip(local_weights, local_eval_val_loss):
                    if score is None:
                        pass
                    else:
                        local_models.append(local_weight)
                        scores.append(score)

            # update global weights
            print(f"Epoch {epoch}, update global model")
            if len(local_weights) != 0:
                for local_weight, score in zip(local_weights, local_eval_val_loss):
                    # BRAIN: aggregate updates using `window`-sized moving average
                    if score is None:
                        pass
                    else:
                        alpha = wma.next(score)

                        if (args.interpol == 'slerp'):
                            global_weights = compose_weight_slerp(
                                global_weights, local_weight, alpha)
                        elif (args.interpol == 'lerp'):  # same as BRAIN
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
                # LERP: TODO: check `weighted_average_weights` working
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
            print(f"Epoch {epoch}, test")
            global_model.to('cuda')
            ppl, ppl_std = test_inference(
                args, global_model, tokenizer, gpu=torch.cuda.current_device())
            print(f"[INFO] Wikitext PPL (wikitext2): {ppl} +- {ppl_std}")
            ppl_collect.append(ppl)

            global_model.to('cpu')
            drifted_model.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            # if 'out of memory' in str(e):
            print(e)
            print(f"[WARNING] {epoch}, stopping training loop.")

            global_model.to('cpu')
            drifted_model.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

            args.epochs = epoch
            break  # or continue
            # else:
            # raise

    # Saving the objects:
    file_name = './save_llm/objects/frain_{}_N{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_D{}_W{}_S{}_TH{}_DR{}_{}_{}_a{}_b{}_c{}_{}.pkl'.\
        format(
            args.epochs, args.num_users, args.frac, args.iid,
            args.local_ep, args.local_bs, args.byzantines, args.score_byzantines,
            args.diff, args.window, args.stale, args.threshold,
            args.drift, args.interpol, args.adaptive,
            args.adaptive_a, args.adaptive_b, args.adaptive_c,
            time.time()
        )

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
    plt.savefig(
        './save_llm/frain_{}_N{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_D{}_W{}_S{}_TH{}_DR{}_{}_{}_a{}_b{}_c{}_ppl.png'.
        format(
            args.epochs, args.num_users, args.frac, args.iid,
            args.local_ep, args.local_bs, args.byzantines, args.score_byzantines,
            args.diff, args.window, args.stale, args.threshold,
            args.drift, args.interpol, args.adaptive,
            args.adaptive_a, args.adaptive_b, args.adaptive_c
        )
    )
