import os
import torch
from torch.optim.adamw import AdamW

from transformers import set_seed, get_cosine_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model, TaskType

from tqdm import tqdm


class OnlyGPU(Exception):
    pass


set_seed(42)


def make_net(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    # model_name: str = "meta-llama/Llama-3.2-1B",
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.1,
    bias="none",
    # target_modules: list[str] = ["q_proj", "v_proj"],
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
    device: str = None,
    # dtype: torch.dtype = torch.float16,
):
    if device:
        device = device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        raise OnlyGPU("only GPU allowed.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        model_max_length=1024,  # TODO
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"The tokenizer.pad_token set as a {tokenizer.eos_token}")
    # tokenizer.pad_token = tokenizer.eos_token

    # 1)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # 2)
    # Random weights
    save_path = f"./src_llm/{model_name}/model_state.pt"
    if not os.path.exists(save_path):
        # Re-init
        model.apply(model._init_weights)
        if hasattr(model, "post_init"):
            model.post_init()
        # save model
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved model state_dict to {save_path}")
    else:
        # Load random weights
        model.load_state_dict(torch.load(
            f"./src_llm/{model_name}/model_state.pt"))

    # # 3)
    # # LoRA
    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,   # fine-tuning mode
    #     r=lora_r,
    #     lora_alpha=lora_alpha,
    #     lora_dropout=lora_dropout,
    #     bias=bias,
    #     target_modules=target_modules,
    # )
    # model = get_peft_model(model, peft_config)

    # ready to train
    model = model.to(device)
    model.train()

    return model, tokenizer


def train(
    model,
    train_loader,
    steps_per_epoch: int,
    # batch_size: int,
    epochs: int,
    warmup_steps: int,
    lr: float = 3e-4,
    device: str = "cuda"
):
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    total_steps = int(steps_per_epoch * epochs)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.to(device)

    # fine-tune loop
    model.train()
    train_loss_collect = []
    for epoch in range(1, epochs+1):
        progress_bar = tqdm(enumerate(train_loader),
                            total=steps_per_epoch, desc=f"Epoch {epoch}/{epochs}")

        running_loss = 0.0
        for step, batch in progress_bar:
            if batch is None:
                continue
            if batch['input_ids'].size(1) == 0:
                continue

            # batch: dict(input_ids, attention_mask, labels)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        avg_train_loss = running_loss / steps_per_epoch
        train_loss_collect.append(avg_train_loss)

    model.eval()
    return train_loss_collect
