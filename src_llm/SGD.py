import time
import pickle

import gc
import torch

from src_llm.impls.options import args_parser
from src_llm.impls.update import LocalUpdate, test_inference

from src_llm.llama.model import make_net


if __name__ == "__main__":
    start_time = time.time()

    args = args_parser()

    model_name = "HuggingFaceTB/SmolLM2-135M"
    # model_name = "meta-llama/Llama-3.2-1B"

    model, tokenizer = make_net(
        model_name=model_name,
        # lora_r=8,
        # lora_alpha=16,
        # lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj"],
    )

    local_update = LocalUpdate(
        args=args,
        tokenizer=tokenizer,
    )

    # TRAIN

    train_loss_collect = []
    val_loss_collect = []
    ppl_collect = []

    # before training
    train_loss_collect.append(None)
    # val_loss = local_update.inference(
    #     model,
    # )
    # print(f"[INFO] Validation Loss: {val_loss:.4f}")
    # val_loss_collect.append(val_loss)
    ppl, ppl_std = test_inference(args, model, tokenizer)
    print(f"[INFO] Wikitext PPL (wikitext2): {ppl} +- {ppl_std}")
    ppl_collect.append(ppl)

    for i in range(args.epochs):
        print("[INFO]", i)

        try:
            # training
            print("[INFO] Start Training ...")
            local_update.update_dataset_train(  # Training dataset setting
                # i  # Seed
                seed=int(time.time() * 1000) & (2**32 - 1)
            )
            state_dict, avg_train_loss = local_update.update_weights(  # Train
                model=model,
                epochs=args.local_ep,
            )
            print(f"[INFO] Average Train Loss: {avg_train_loss:.4f}")
            train_loss_collect.append(avg_train_loss)

            # # validation
            # model.load_state_dict(state_dict)
            # val_loss = local_update.inference(
            #     model,
            # )
            # print(f"[INFO] Validation Loss: {val_loss:.4f}")
            # val_loss_collect.append(val_loss)

            # eval (testing)
            ppl, ppl_std = test_inference(args, model, tokenizer)
            print(f"[INFO] Wikitext PPL (wikitext2): {ppl} +- {ppl_std}")
            ppl_collect.append(ppl)

            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:  # TODO: print error msg
            # if 'out of memory' in str(e):
            print(f"[WARNING] CUDA OOM at epoch {i}, stopping training loop.")
            torch.cuda.empty_cache()
            args.epochs = i
            break  # or continue
            # else:
            # raise

    # print(model)

    # Saving the objects:
    file_name = './save_llm/objects/nn_{}_{}.pkl'.format(
        args.epochs, time.time())
    with open(file_name, 'wb') as f:
        pickle.dump(
            [
                # train_loss_collect,
                # val_loss_collect,
                ppl_collect
            ],
            f
        )
    print(len(ppl_collect))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Training Loss
    plt.figure()
    plt.title('Loss vs Communication rounds')
    plt.plot(range(len(train_loss_collect)), train_loss_collect, color='r')
    plt.ylabel('Training Loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save_llm/nn_{}_trainloss.png'.format(args.epochs))

    # # Plot Validation Loss
    # plt.figure()
    # plt.title('Loss vs Communication rounds')
    # plt.plot(range(len(val_loss_collect)), val_loss_collect, color='r')
    # plt.ylabel('Val Loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('./save_llm/nn_{}_valloss.png'.format(args.epochs))

    # Plot Test PPL
    plt.figure()
    plt.title('PPL vs Communication rounds (wikitext)')
    plt.plot(range(len(ppl_collect)), ppl_collect, color='k')
    plt.ylabel('Test PPL')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save_llm/nn_{}_testppl.png'.format(args.epochs))
