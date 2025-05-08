import argparse
import torch


from model import make_net
from src_llm.impls.update import LocalUpdate, test_inference


if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--local_bs",
            type=int,
            default=16,
            help="Training batch size"
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            help="Number of training epochs"
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=3e-4,
            help="Learning rate"
        )
        parser.add_argument(
            "--gpu",
            type=int,
            default=0,
            help="GPU device index to use"
        )
        parser.add_argument(
            "--num_train_steps",
            type=int,
            default=100,
            help="Number of training samples to stream"
        )
        parser.add_argument(
            "--num_val_steps",
            type=int,
            default=10,
            help="Number of validation samples to stream"
        )
        parser.add_argument(
            "--eval_bs",
            type=int,
            default=16,
            help="Batch size for evaluation"
        )
        parser.add_argument(
            "--num_eval_steps",
            type=int,
            default=100,
            help="Amount of data to sample during evaluation (token limit)"
        )
        return parser.parse_args()

    args = get_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # model_name = "meta-llama/Llama-3.2-1B"
    model_name = "HuggingFaceTB/SmolLM2-135M"

    model, tokenizer = make_net(
        model_name=model_name,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        device=device.type
    )

    local_update = LocalUpdate(
        args=args,
        tokenizer=tokenizer,
        gpu=args.gpu
    )

    print("[INFO] Start Training ...")
    state_dict, avg_train_loss = local_update.update_weights(
        model=model,
        epochs=args.epochs
    )
    print(
        f"[INFO] Finished Training. Average Train Loss: {avg_train_loss:.4f}")

    model.load_state_dict(state_dict)
    val_loss = local_update.inference(model)
    print(f"[INFO] Validation Loss: {val_loss:.4f}")

    ppl, ppl_std = test_inference(args, model, tokenizer, gpu=args.gpu)
    print(f"[INFO] Wikitext PPL: {ppl} +- {ppl_std}")

    print(model)
