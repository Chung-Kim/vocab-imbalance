#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1


ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=PRETRAIN accelerate launch -m \
    --config_file accelerate/ds3.yaml \
    scripts.run_alignment \
    recipes/samples/14m.yaml