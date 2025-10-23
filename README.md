# Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training 

Woojin Chung and Jeonghoon Kim


Install 

```bash
pip install -r requirements.txt
```

20 Experiments needed to run (24K, 49K, 98K, 196K) * (7.5e-5, 1.5e-4, 1.2e-3, 2.4e-3) + (24K_embdtied, 49K_embdtied, 98K_embdtied, 196K_embdtied)

run code in token-frequency directory

--24K models--

```bash
bash run/24k_7.5e-5.sh, bash run/24k_1.5e-4.sh, bash run/24k_1.2e-3.sh, bash run/24k_2.4e-3.sh
```

--tied models--
```bash
bash run/24k_embdtied.sh, bash run/49k_embdtied.sh, bash run/98k_embdtied.sh, bash run/196k_embdtied.sh
```

--450M models--
```bash
bash run/24k_450M.sh, bash run/49k_450M.sh, bash run/98k_450M.sh, bash run/196k_450M.sh
```

Also set local_dir and wandb in 24k.yaml ~ 196k.yaml



