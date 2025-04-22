# token-frequency


Install 

```bash
pip install -r requirements.txt
```

4 Experiments needed to run (24K, 49K, 98K, 196K) 

run code in token-frequency directory
```bash
bash run/train_24k.sh, bash run/train_24k.sh, bash run/train_24k.sh, bash run/train_24k.sh
```

Also set local_dir and wandb in 24k.yaml ~ 196k.yaml

config: num_layer:24, num_attention_head:24, hidden_size:1536, intermediate_size: 4096 ==> 8 times bigger non-embedding size than 85M (non-embed) experiments

