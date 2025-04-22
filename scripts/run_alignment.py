
import os
import logging
import sys
from typing import Any, Dict

import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    set_seed,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import datasets
from datasets import (
    load_dataset,
    concatenate_datasets
)
from trl import (
    SFTTrainer,
    SFTConfig,
    ORPOTrainer,
    ORPOConfig,
    RewardConfig,
    RewardTrainer,
    DPOTrainer,
    DPOConfig
)
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from modules import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    map_chat_template_by_task,
    initialize_reward_model_head,
    DEFAULT_CHAT_TEMPLATE,
    initialize_model,
    apply_liger_kernel_to_llama_with_z_loss,
    initialize_model_auxloss
    
)


logger = logging.getLogger(__name__)

class UpdateStepCallback(TrainerCallback):
    """
    A callback to update the global_step in the model during training.
    """
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        if hasattr(model, "set_step"):
            model.set_step(state.global_step)


def main(model_args, data_args, training_args, training_type: str, loss_type_test: str, base_trainer: Trainer):
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load dataset
    dataset = load_dataset(data_args.dataset_name, cache_dir=model_args.cache_dir, split=data_args.dataset_split, data_files=data_args.dataset_file_list, cache_dir=data_args.cache_dir)
    #dataset = load_dataset(data_args.dataset_name, cache_dir=model_args.cache_dir, split=data_args.dataset_split)
    if hasattr(dataset, "shuffle"):
        pass
    else:
        dataset = concatenate_datasets(dataset)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    if training_type.lower() in ['pretrain', 'proposed'] and loss_type_test.lower() == 'zloss':
        apply_liger_kernel_to_llama_with_z_loss(cross_entropy=False, fused_linear_cross_entropy=True)

    # Load Model
    model_wrapper = AutoLigerKernelForCausalLM if model_args.use_liger_lm else AutoModelForCausalLM
    if training_type.lower() != 'rm':
        if training_type == "PRETRAIN":
            model = initialize_model(model_args.attn_implementation, torch_dtype, tokenizer)
        if training_type == "PROPOSED":
            model = initialize_model_auxloss(model_args.attn_implementation, torch_dtype, tokenizer)
        if training_type.lower() == 'dpo':
            ref_model = model_wrapper.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                attn_implementation=model_args.attn_implementation,
                torch_dtype=torch_dtype,
                use_cache=False if training_args.gradient_checkpointing else True
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            num_labels=1
        )
        model, tokenizer = initialize_reward_model_head(model=model, tokenizer=tokenizer)
        ref_model = None

    ### Set chat template
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        logger.info(f"Chat template is set to Zephyr template as Tokenizer did not have one.")
    elif data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    else:
        pass

    # Preprocess dataset
    column_names = list(dataset.features)
    if training_type not in ("PRETRAIN", "PROPOSED"):
        preprocessed_dataset = dataset.map(
            map_chat_template_by_task,
            fn_kwargs={
                "tokenizer": tokenizer,
                "training_type": training_type,
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },        
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            batched=True if training_type in ["RM", "PRETRAIN"] else False,
            desc="Formatting comparisons with prompt template",
        )
    else:
        preprocessed_dataset = dataset
        preprocessed_dataset.set_format("torch", columns=["input_ids"], output_all_columns=True)

    # Filter long instances if ORPO or LINEAR
    if training_type.lower in ['orpo', 'linear']:
        if training_args.max_prompt_length is not None:
            unfiltered_train_samples = len(preprocessed_dataset)

            def filter_fn(sample: Dict[str, Any]) -> Dict[str, Any]:
                prompt_length = tokenizer(
                    sample["text_prompt"],
                    return_tensors="pt",
                )[
                    "input_ids"
                ].size(dim=-1)

                return prompt_length < training_args.max_prompt_length

            preprocessed_dataset = preprocessed_dataset.filter(
                filter_fn,
                desc="Filtering out the samples where len(text_prompt) > max_prompt_length",
                num_proc=data_args.preprocessing_num_workers
            )

            filtered_train_samples = unfiltered_train_samples - len(preprocessed_dataset)
            logger.info(
                f"Filtered out {filtered_train_samples} training samples out of the {unfiltered_train_samples} samples."
            )
    else:
        pass

    if training_type.lower() in ['dpo', 'orpo', 'linear']:
        preprocessed_dataset = preprocessed_dataset.rename_columns(
            {
                "text_prompt": "prompt",
                "text_chosen": "chosen",
                "text_rejected": "rejected",
            }
        )

    # Print sample items
    # print_sample_items(data=preprocessed_dataset, logger=logger, training_type=training_type, sample_num=2)

    ########################
    # Initialize the Trainer
    ########################
    print(">>> List of Dataset Features:", list(preprocessed_dataset.features))

    if training_args.run_name is None:
        training_args.run_name = training_args.hub_model_id

    callbacks = [transformers.integrations.MLflowCallback()] if "/shared/public" in model_args.model_name_or_path else None
    if training_type.lower() in ['orpo', 'rm', 'sft', 'linear']:
        trainer = base_trainer(
            model,
            args=training_args,
            train_dataset=preprocessed_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
    elif training_type.lower() in ['pretrain', 'proposed']:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're not using masked language modeling here
        )
        trainer = base_trainer(
            model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=preprocessed_dataset,
            data_collator=data_collator,
            callbacks=[UpdateStepCallback()],
        )
    else:
        trainer = base_trainer(
            model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=preprocessed_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )


    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(preprocessed_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Push to hub
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Select proper Config and Trainer class
    training_type = os.getenv("TRAINING_TYPE", "ORPO")
    loss_type_test = os.getenv("LOSS_TYPE", "TEMP")
    if training_type == "ORPO":
        config_type = ORPOConfig
        base_trainer = ORPOTrainer
    elif training_type == 'SFT':
        config_type = SFTConfig
        base_trainer = SFTTrainer
    elif training_type == 'DPO':
        config_type = DPOConfig
        base_trainer = DPOTrainer
    elif training_type == 'RM':
        config_type = RewardConfig
        base_trainer = RewardTrainer
    elif training_type == 'PRETRAIN':
        config_type = TrainingArguments
        base_trainer = Trainer
    elif training_type == 'PROPOSED':
        config_type = TrainingArguments
        base_trainer = Trainer
    else:
        raise Exception("Please check the training method. You should set it to one of: ORPO, SFT, RM, LINEAR, PRETRAIN")

    # Parse arguments
    parser = H4ArgumentParser((ModelArguments, DataArguments, config_type))
    model_args, data_args, training_args = parser.parse()

    # Set up WandB is needed
    if data_args.wandb_entity is not None and data_args.wandb_project is not None:
        os.environ["WANDB_ENTITY"] = data_args.wandb_entity
        os.environ["WANDB_PROJECT"] = data_args.wandb_project

    # Start training
    main(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        training_type=training_type,
        loss_type_test=loss_type_test,
        base_trainer=base_trainer
    )
