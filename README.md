# [NeurIPS 2025] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training 

Woojin Chung and Jeonghoon Kim

This is the official implementation for our NeurIPS 2025 paper: "Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training".


# Hugging Face Hub


All models and datasets reported in the paper are available in our Hugging Face collection [Vocab Frequency Imbalance](https://huggingface.co/collections/gartland/neurips-2025-vocabulary-frequency-imbalance).

**Install**


```bash
pip install -r requirements.txt
```

# Pre-training 



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

# Citation

```bash
@article{DBLP:journals/corr/abs-2508-15390,
  author       = {Woojin Chung and
                  Jeonghoon Kim},
  title        = {Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training},
  journal      = {CoRR},
  volume       = {abs/2508.15390},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2508.15390},
  doi          = {10.48550/ARXIV.2508.15390},
  eprinttype    = {arXiv},
  eprint       = {2508.15390},
  timestamp    = {Thu, 18 Sep 2025 17:28:53 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2508-15390.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```



