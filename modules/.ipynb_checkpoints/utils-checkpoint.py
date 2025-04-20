
# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import random
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from typing import Any, List, Literal
import json


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"



def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})

def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False

def map_chat_template_by_task(
    example,
    tokenizer,
    training_type: Literal["SFT", "RM", "ORPO", "DPO", "LINEAR", "PRETRAIN", "PROPOSED"],
    auto_insert_empty_system_msg: bool = False,
):
    ### TODO: Handle chat templates with inherent errors
    if training_type.lower() == "sft":
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    elif training_type.lower() in ['pretrain','proposed']:
        example['input_ids'] = torch.tensor(example['input_ids'])
        example['labels'] = torch.tensor(example['input_ids'])
    elif training_type.lower() == "rm":
        example["input_ids_chosen"] = []
        example["attention_mask_chosen"] = []
        example["input_ids_rejected"] = []
        example["attention_mask_rejected"] = []
        
        for chosen, rejected in zip(example["chosen"], example["rejected"]):
            tokenized_chosen_ = tokenizer.apply_chat_template(chosen, add_generation_prompt=False, tokenize=False)
            tokenized_rejected_ = tokenizer.apply_chat_template(rejected, add_generation_prompt=False, tokenize=False)
            
            tokenized_chosen = tokenizer(tokenized_chosen_, add_special_tokens=False)
            tokenized_rejected = tokenizer(tokenized_rejected_, add_special_tokens=False)

            example["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            example["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            example["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            example["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    elif training_type.lower() in ['dpo', 'orpo', 'linear']:
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{training_type}` training_type! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)
            
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

            if tokenizer.bos_token:
                if example["text_chosen"].startswith(tokenizer.bos_token): 
                    example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):] 
                if example["text_rejected"].startswith(tokenizer.bos_token): 
                    example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):] 
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{training_type}` training_type! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"training_type {training_type} not supported, please ensure that the provided training_type is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example

def print_sample_items(
    data,
    logger,
    training_type: str,
    sample_num: int = 3,
):
    if training_type.lower() in ["orpo", "dpo", "linear"]:
        for index in random.sample(range(len(data)), sample_num):
            logger.info(f"Prompt sample {index} of the raw training set:\n\n{data[index]['prompt']}")
            logger.info(f"Chosen sample {index} of the raw training set:\n\n{data[index]['chosen']}")
            logger.info(f"Rejected sample {index} of the raw training set:\n\n{data[index]['rejected']}")
    elif training_type.lower() == "sft":
        for index in random.sample(range(len(data)), sample_num):
            logger.info(f"Sample {index} of the processed training set:\n\n{data[index]['text']}")
    elif training_type.lower() == 'rm':
        pass
    else:
        raise Exception("Check the training type.")

def get_batches(items, batch_size):
    num_batches = (len(items) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(items))
        batch = items[start_index:end_index]
        batches.append(batch)

    return batches


def initialize_reward_model_head(model: AutoModel, tokenizer: AutoTokenizer):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.truncation_side = "left"

    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    
    model.score = nn.Linear(model.config.hidden_size, 1, bias=False)

    print(">>> Classification head initialized to with normal distribution.: ", model.score.weight.size())
    nn.init.normal_(model.score.weight, mean=0.0, std=1/np.sqrt(model.config.hidden_size+1))

    return model, tokenizer

def load_config(file_path="config.json"):
    """Load model configuration from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def initialize_model(attn_implementation, torch_dtype, tokenizer, config_path="./recipes/config/config.json"):
    """Initialize model with configuration loaded from a JSON file."""
    config_dict = load_config(config_path)
    config_dict["vocab_size"] = tokenizer.vocab_size  # Ensure vocab_size is updated

    # Create config object
    config = LlamaConfig.from_dict(config_dict)

    # Initialize model with custom config
    model = AutoModelForCausalLM.from_config(
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    return model

class OutputEmbeddingSelectiveUpdate(LlamaForCausalLM):
    def __init__(self, config, vocab_size, training_args, lambda_global=0.01, lambda_per_token=0.01, lambda_norm_gap=0.01, lambda_norm_penalty=0.01):
        super().__init__(config)
        self.lambda_global = lambda_global  # Regularization strength for global mean centering
        self.lambda_per_token = lambda_per_token  # Regularization strength for per-token mean bias
        self.lambda_norm_gap = lambda_norm_gap  # Regularization for norm gap between frequent and infrequent tokens
        self.lambda_norm_penalty = lambda_norm_penalty  # Regularization for overall token norm growth
    
    def compute_embedding_mean(self):
        """
        Computes the mean of the embedding weights.
        """
        return self.get_input_embeddings().weight.mean(dim=0)
    
    def compute_per_token_mean_bias(self):
        """
        Computes the per-token mean bias.
        """
        return self.get_input_embeddings().weight.mean(dim=-1)
    
    def compute_embedding_norms(self):
        """
        Computes the L2 norm of each token embedding.
        """
        return torch.norm(self.get_input_embeddings().weight, p=2, dim=-1)
    
    def compute_norm_gap_penalty(self):
        """
        Penalizes the standard deviation of token embedding norms to reduce norm gap between frequent and infrequent tokens.
        """
        norms = self.compute_embedding_norms()
        return self.lambda_norm_gap * torch.std(norms)
    
    def compute_norm_penalty(self):
        """
        Penalizes excessive token norm growth by minimizing the mean norm value.
        """
        norms = self.compute_embedding_norms()
        return self.lambda_norm_penalty * torch.mean(norms)
    
    def forward(self, input_ids, *args, **kwargs):
        # Forward pass
        outputs = super().forward(input_ids=input_ids, *args, **kwargs)
        
        # Compute zero-mean regularization loss
        embedding_mean = self.compute_embedding_mean()
        global_reg_loss = self.lambda_global * torch.norm(embedding_mean, p=2) ** 2
        
        # Compute per-token mean bias regularization loss
        per_token_bias = self.compute_per_token_mean_bias()
        per_token_reg_loss = self.lambda_per_token * torch.mean(torch.abs(per_token_bias))
        
        # Compute norm gap and norm penalty losses
        norm_gap_loss = self.compute_norm_gap_penalty()
        norm_penalty_loss = self.compute_norm_penalty()
        
        # Add regularization terms to total loss if available
        if outputs.loss is not None:
            outputs.loss += global_reg_loss + per_token_reg_loss + norm_gap_loss + norm_penalty_loss
        
        return outputs

    @classmethod
    def from_config(cls, config, vocab_size, training_args, torch_dtype=None, attn_implementation=None, lambda_global=0.01, lambda_per_token=0.01, lambda_norm_gap=0.01, lambda_norm_penalty=0.01):
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        return cls(model.config, vocab_size, training_args, lambda_global=lambda_global, lambda_per_token=lambda_per_token, lambda_norm_gap=lambda_norm_gap, lambda_norm_penalty=lambda_norm_penalty)


def initialize_model_auxloss(attn_implementation, torch_dtype, tokenizer, config_path="./recipes/config/config_auxloss.json"):
    """Initialize model with configuration loaded from a JSON file."""
    config_dict = load_config(config_path)
    config_dict["vocab_size"] = tokenizer.vocab_size  # Ensure vocab_size is updated

    # Create config object
    config = LlamaConfig.from_dict(config_dict)

    # Initialize model with custom config
    model = OutputEmbeddingSelectiveUpdate.from_config(config, config.vocab_size, training_args=training_args ,torch_dtype=torch_dtype,
        attn_implementation=attn_implementation)

    return model
    


