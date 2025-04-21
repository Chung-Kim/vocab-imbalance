# token-frequency


Install 

```bash
pip install -r requirements.txt
```

4 Experiments needed to run (24K, 49K, 98K, 196K) 

run 
```bash
bash run/train.sh
```


config: num_layer:24, num_attention_head:24, hidden_size:1536, intermediate_size: 4096 ==> 8 times bigger non-embedding size than 85M (non-embed) experiments

only change 14m.yaml file

1. tokenizer_name_or_path (24K: gartland/finewebedu-24K-tokenizer, 49K: gartland/finewebedu-49K-tokenizer, 98K: gartland/finewebedu-98K-tokenizer 196K: gartland/finewebedu-196K-tokenizer)
2. dataset_name (24K: gartland/finewebedu-24K-tokenized, 49K: gartland/finewebedu-49K-tokenized, 98K: gartland/finewebedu-98K-tokenized, 196K: gartland/finewebedu-196K-tokenized)
3. gradient accumulation step (24K:4, 49K:8, 98K:8, 196K:16)
4. per_device_train_batch_size (24K:8, 49K:4, 98K:4, 196K:2)
5. wandb settings

