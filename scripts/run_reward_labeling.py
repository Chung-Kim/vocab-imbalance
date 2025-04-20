
"""
Batch Inference for Reward Labeling with accelerator
"""

import torch
import logging
import os
import gc
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import gather_object

from modules import (
    RewardArguments, 
    DataArguments, 
    H4ArgumentParser
)

logger = logging.getLogger(__name__)

def main():
    parser = H4ArgumentParser((DataArguments, RewardArguments))
    data_args, reward_args = parser.parse()

    # Prepare accelerator
    distributed_state = Accelerator()

    # Load dataset and preprocess
    data = load_dataset(data_args.dataset_name, split=data_args.dataset_split, cache_dir=reward_args.cache_dir).shard(num_shards=1000, index=0)
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_args.model_name_or_path, 
        trust_remote_code=True, 
        cache_dir=reward_args.cache_dir,
        attn_implementation=reward_args.attn_implementation,
        torch_dtype=torch.bfloat16
    ).to(device=distributed_state.device)
    
    tokenizer = AutoTokenizer.from_pretrained(reward_args.model_name_or_path, use_fast=True, cache_dir=reward_args.cache_dir)

    # Run distributed inference
    input_sequences = []
    output_scores = []

    with distributed_state.split_between_processes(data[data_args.text_column]) as inputs:
        score_temp = []

        logger.info("Tokenizing inputs sequences...")
        chat_inputs = [
            tokenizer.apply_chat_template(item, add_generation_prompt=False, tokenize=True, return_tensors='pt') for item in inputs
        ]

        if distributed_state.is_main_process:
            pbar = tqdm(total=len(chat_inputs), desc=f"[Main Process] Labeling with {reward_args.model_name_or_path.split('/')[-1]}...")

        for input_ids in chat_inputs:
            with torch.no_grad():
                scores = model(
                    input_ids=input_ids.to(model.device)
                ).logits.item()

            score_temp.append(scores)

            del input_ids, scores
            torch.cuda.empty_cache()

            if distributed_state.is_main_process:
                pbar.update(1)
        
        gc.collect()

        if distributed_state.is_main_process:
            pbar.close()

    distributed_state.wait_for_everyone()

    inputs = gather_object(inputs)
    score_temp = gather_object(score_temp)

    input_sequences.extend(inputs)
    output_scores.extend(score_temp)

    torch.cuda.empty_cache()
    gc.collect()


    # Push final dataset
    if distributed_state.is_main_process:
        final_data = []

        for chat, score in zip(input_sequences, output_scores):
            final_data.append({
                'messages': chat,
                'score': score
            })
        
        final_data_to_push = Dataset.from_list(final_data[:len(data)])

        if reward_args.save_directory:
            final_data_to_push.save_to_disk(reward_args.save_directory)
        else:
            final_data_to_push.push_to_hub(
                repo_id=f'linkedin-xfact/RM-labels-{data_args.dataset_name.split("/")[-1]}-{reward_args.model_name_or_path.split("/")[-1]}',
                private=True,
            )


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    main()
