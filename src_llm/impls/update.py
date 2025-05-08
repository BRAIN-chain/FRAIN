import time
from math import ceil
import copy
import torch
from torch.utils.data import DataLoader

from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import AutoTokenizer

from lm_eval.models import huggingface
from lm_eval import simple_evaluate


from src_llm.llama.model import train


class LocalUpdate(object):
    def __init__(
        self,
        args,
        tokenizer,
        gpu=0
    ):
        self.args = args
        self.device = torch.device(
            f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

        self.tokenizer = tokenizer

        def tokenize_function_text(examples):
            return self.tokenizer(
                examples['text'],
                return_special_tokens_mask=True,
                max_length=1024,  # TODO
                truncation=True
            )

        data_collator_lm = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # causal LM
            return_tensors='pt'
        )

        # # === train set ===
        # Set later (via `update_dataset_train`)

        # === validation set ===
        ds = load_dataset(
            'Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation', streaming=False
        )
        ds = ds.shuffle(seed=int(time.time() * 1000) & (2**32 - 1))
        self.dataset_val = ds.select(range(self.args.num_val_steps))
        tokenized_val = self.dataset_val.map(
            tokenize_function_text,
            batched=True,
            # remove_columns=['text']
            remove_columns=self.dataset_val.features.keys()
        )
        self.val_dataloader = DataLoader(
            tokenized_val,
            batch_size=self.args.local_bs,
            collate_fn=data_collator_lm
        )

    # FL
    def set_dataset_train(self, skip_num, take_num, seed=42):
        def tokenize_function_text(examples):
            return self.tokenizer(
                examples['text'],
                return_special_tokens_mask=True,
                max_length=1024,  # TODO
                truncation=True
            )

        data_collator_lm = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # causal LM
            return_tensors='pt'
        )

        # === train set ===
        ds = load_dataset(
            'Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', streaming=False
        ).skip(skip_num).take(take_num)
        # ds = ds.shuffle(seed=seed)
        # self.dataset_train = ds.select(
        #     # range(self.args.num_train_steps)
        #     range(int(take_num * self.args.local_sub_ep))
        # )
        self.dataset_train = ds.shuffle(seed=seed)
        tokenized_train = self.dataset_train.map(
            tokenize_function_text,
            batched=True,
            # remove_columns=['text']
            remove_columns=self.dataset_train.features.keys()
        )
        # train DataLoader
        self.train_dataloader = DataLoader(
            tokenized_train,
            batch_size=self.args.local_bs,
            collate_fn=data_collator_lm
        )

    # SGD
    def update_dataset_train(self, seed):
        def tokenize_function_text(examples):
            return self.tokenizer(
                examples['text'],
                return_special_tokens_mask=True,
                max_length=1024,  # TODO
                truncation=True
            )

        data_collator_lm = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # causal LM
            return_tensors='pt'
        )

        # === train set ===
        ds = load_dataset(
            'Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', streaming=False
        )
        ds = ds.shuffle(seed=seed)
        self.dataset_train = ds.select(
            range(self.args.num_train_steps)
        )
        tokenized_train = self.dataset_train.map(
            tokenize_function_text,
            batched=True,
            # remove_columns=['text']
            remove_columns=self.dataset_train.features.keys()
        )
        # train DataLoader
        self.train_dataloader = DataLoader(
            tokenized_train,
            batch_size=self.args.local_bs,
            collate_fn=data_collator_lm
        )

    def update_weights(self, model, epochs=1, global_round=None, verbose=0):
        lr = self.args.lr
        # warmup_steps = getattr(self.args, 'warmup_steps', 100)
        warmup_steps = len(self.train_dataloader) * epochs // 50  # 2%

        train_loss_collect = train(
            model=model,
            train_loader=self.train_dataloader,
            # steps_per_epoch=ceil(self.args.num_train_steps / self.args.local_bs),
            steps_per_epoch=len(self.train_dataloader),
            epochs=epochs,
            warmup_steps=warmup_steps,
            lr=lr,
            device=self.device
        )
        avg_loss = sum(train_loss_collect) / len(train_loss_collect)
        if verbose == 0:
            return model.state_dict(), avg_loss
        else:
            return model.state_dict(), train_loss_collect

    def inference(self, model, verbose=0):
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        if verbose == 0:
            return avg_loss
        else:
            return total_loss, num_batches


class ByzantineLocalUpdate(LocalUpdate):
    def update_weights(self, model, epochs=1, global_round=None, verbose=0):
        w_t = copy.deepcopy(model.state_dict())
        for key in w_t.keys():
            # w_t[key] = torch.zeros_like(w_t[key])  # Nullifier
            w_t[key] = torch.randn_like(w_t[key])  # Randomizer
        return w_t, None


def test_inference(args, model, tokenizer, gpu=0):
    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model.eval()

    task_name = 'wikitext'
    batch_size = getattr(args, 'eval_bs', 128)

    lm = huggingface.HFLM(
        pretrained=model,
        backend='causal',
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=1024,  # TODO
    )

    limit = args.num_eval_steps
    outputs = simple_evaluate(
        model=lm,
        tasks=[task_name],
        limit=limit
    )

    output = outputs['results'][task_name]
    ppl_word_perplexity = output['word_perplexity,none']
    ppl_word_perplexity_stderr = output['word_perplexity_stderr,none']

    return ppl_word_perplexity, ppl_word_perplexity_stderr
