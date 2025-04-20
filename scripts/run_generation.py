
"""
Efficient generation with vLLM support
"""

import logging
import os
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams

from modules import GenerationArguments, DataArguments, H4ArgumentParser


logger = logging.getLogger(__name__)

def main(current_iteration: int = 1):
    parser = H4ArgumentParser((DataArguments, GenerationArguments))
    data_args, gen_args = parser.parse()

    # Load dataset and preprocess
    data = load_dataset(data_args.dataset_name, split=data_args.dataset_split, cache_dir=gen_args.cache_dir)  
    prompts_list = data[data_args.text_column]

    # Load model and tokenizer
    llm = LLM(model=gen_args.model_name_or_path,
              max_model_len=gen_args.max_model_len,
              tensor_parallel_size=gen_args.tensor_parallel_size,
              gpu_memory_utilization=gen_args.gpu_memory_utilization,
              download_dir=gen_args.cache_dir)
    tokenizer = llm.get_tokenizer()

    # Set sampling params
    sampling_params = SamplingParams(temperature=gen_args.temperature,
                                     max_tokens=gen_args.max_new_tokens,
                                     top_p=gen_args.top_p,
                                     truncate_prompt_tokens=gen_args.max_model_len - gen_args.max_new_tokens,
                                     seed=gen_args.seed)

    # Apply chat templates
    if gen_args.system_prompt is not None and "mistral" not in gen_args.model_name_or_path.lower():
        prompts = [
            tokenizer.apply_chat_template(
                [
                {
                    'role': 'system',
                    'content': gen_args.system_prompt
                },
                {
                    'role': 'user', 
                    'content': instruction
                }
                ], add_generation_prompt=True, tokenize=False
            ) for instruction in prompts_list
        ]
    else:
        prompts = [
            tokenizer.apply_chat_template(
                [{
                    'role': 'user', 
                    'content': gen_args.system_prompt + instruction
                }
                ], add_generation_prompt=True, tokenize=False
            ) for instruction in prompts_list
        ]
    print(f">>> Example:\n\n\n{prompts[-1]}")
    
    # Generate responses
    outputs = llm.generate(prompts, sampling_params)

    output_data = []

    for i, output in enumerate(outputs):
        generated_text = [te.text for te in output.outputs] if gen_args.num_return_sequences > 1 else output.outputs[0].text
        output_data.append({
            'prompt': prompts_list[i],
            'responses': generated_text,
        })

    data_to_push = Dataset.from_list(output_data)

    if gen_args.save_directory:
        data_to_push.save_to_disk(gen_args.save_directory)
    else:
        data_to_push.push_to_hub(
            repo_id=f'linkedin-xfact/Gen-{data_args.dataset_name.split("/")[-1]}-{gen_args.model_name_or_path.split("/")[-1]}',
            private=True,
        )


if __name__ == "__main__":
    main()
