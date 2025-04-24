import os
import shutil
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from huggingface_hub import create_repo, Repository
import torch.multiprocessing as mp
from itertools import chain

# 1. Load the fineweb sample-10BT dataset
# Adjust the dataset identifier and split if needed.
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")


# 2. Load the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("gartland/finewebedu-49K-tokenizer")

tokenizer.model_max_length = 2049
print(tokenizer.model_max_length)

num_proc = mp.cpu_count()

def group_texts(examples):
    tokenized_inputs = tokenizer(
       examples["text"], return_special_tokens_mask=False, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

tokenized_datasets = dataset.map(group_texts, batched=True, remove_columns=["text"], num_proc=num_proc)
tokenized_datasets = tokenized_datasets.remove_columns([col for col in tokenized_datasets.column_names if col != "input_ids"])

def group_text(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }

    #result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = tokenized_datasets.map(group_text, batched=True, num_proc=num_proc)
tokenized_datasets = tokenized_datasets.remove_columns([col for col in tokenized_datasets.column_names if col != "input_ids"])
# shuffle dataset
#tokenized_datasets = tokenized_datasets.shuffle(seed=5)

# 4. Save the tokenized dataset as Parquet files.
output_dir = "finewebedu-49K-tokenized"
os.makedirs(output_dir, exist_ok=True)

shard_size = len(tokenized_datasets) // 50  # for 10 shards, adjust as needed
for i in range(50):
    start_idx = i * shard_size
    end_idx = (i + 1) * shard_size
    shard = tokenized_datasets.select(range(start_idx, end_idx))
    shard.to_parquet(f"finewebedu-49K-tokenized/train-{i:02d}-of-10.parquet")

print(f"the dataset contains in total {len(tokenized_datasets)*tokenizer.model_max_length} tokens")
print(len(tokenized_datasets))


