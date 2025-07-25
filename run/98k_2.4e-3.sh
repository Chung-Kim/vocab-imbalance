#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1


ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=PROPOSED accelerate launch -m \
    scripts.run_alignment \
    recipes/samples/98k_2.4e-3.yaml
