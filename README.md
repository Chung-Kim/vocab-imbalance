# [NeurIPS 2025] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training 

Woojin Chung and Jeonghoon Kim

This is the official implementation for our NeurIPS 2025 paper: "Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training".


# Hugging Face Hub


All models and datasets reported in the paper are available in our Hugging Face space.

**Install**


```bash
pip install -r requirements.txt
```

# Pre-training the Model



This codebase is built on LinkedIn's Liger Kernel. For API details, see the [Liger Kernel repository](https://github.com/linkedin/Liger-Kernel).

**Run scripts**
- Put your launch scripts in the run directory, then execute, e.g.:
```bash
bash run/24k_7.5e-5.sh
```

# Custom configurations
For a custom model, create a config.json and place it under recipes/config/.
Put your training YAML (data paths, hyperparameters, etc.) under recipes/samples/.
Ensure your run script passes the path to this YAML at the end of the command.




