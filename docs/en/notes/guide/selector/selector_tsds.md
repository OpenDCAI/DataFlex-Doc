---
title: selector_tsds
createTime: 2025/11/01 21:36:21
permalink: /en/guide/im5q9cd2/
icon: tdesign:cat
---


# TSDS Selector Guide

This document explains how to use the **TSDS Selector** (Data Selection for Task‑Specific Model Finetuning) in the **DataFlex** framework to perform **dynamic training data selection** during supervised finetuning (SFT), balancing **representative density** and **topological diversity** to improve generalization.

---

## 1. Method Overview

The core idea of **TSDS** is:

* Further encode **already tokenized** samples into **sentence embeddings** (e.g., 512‑dim).
* Perform **nearest‑neighbor search & kernel density estimation (KDE)** in the embedding space to obtain each sample’s representativeness score.
* Incorporate **topological diversity** (avoid only picking clusters), and trade off density vs. diversity via the coefficient `alpha`.

> Intuition: **Higher density** ⇒ more “typical/representative” samples; **higher diversity** ⇒ broader coverage and less redundancy.

### Scoring Formulation

Let the sentence embedding of a sample be $e_i$, and let its $K$ nearest neighbors be $\mathcal{N}_K(i)$.


1. **Kernel Density Estimation (KDE):**
   $$
   \text{density}(i)
   = \frac{1}{K} \sum_{j\in \mathcal{N}_K(i)}
   \exp!\left(-\frac{\lVert e_i - e_j \rVert^2}{2\sigma^2}\right)
   $$

2. **Diversity (simple implementation via de‑dup penalty / marginal gain):**
   $$
   \text{diversity}(i)\ \propto
   \min_{j\in S} \lVert e_i - e_j \rVert,\quad
   S=\text{selected set}
   $$

3. **Combined Score:**
   $$
   \text{score}(i)
   = \alpha, \text{density}(i)

   * (1-\alpha), \text{diversity}(i)
   $$

> In practice, `kde_K` (neighbors used by KDE) and `max_K` (overall NN search limit) can differ. `C` can be used as a selection ratio/threshold or other control term depending on the implementation.

---

## 2. Environment & Dependencies

```bash
# DataFlex (recommended: editable install)
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .

# Common training/inference dependencies (as needed)
pip install llamafactory

# TSDS extras (vector search & progress bars)
pip install faiss-cpu tqdm
```

---

## 3. Selector Registration & Initialization

Register a custom TSDS selector component:

```python
from dataflex.selectors import Selector, register_selector

@register_selector("tsds")
class TsdsSelector(Selector):
    """Topological & Statistical Density Selector"""
    def __init__(
        self,
        dataset,
        eval_dataset,
        accelerator,
        data_collator,
        cache_dir,
        seed: int = 42,
        max_K: int = 128,
        kde_K: int = 64,
        sigma: float = 1.0,
        alpha: float = 0.5,
        C: float = 10.0,
        sample_size: int = 1000,
        model_name: str = "/home/lianghao/yry/TSDS/bert_chinese"  # sentence encoder path
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        
```
**TODO: Replace `model_name` with your local encoder path. Using the placeholder will raise an error.**

> **Note:** `model_name` is used to encode **tokenized** samples into **sentence embeddings** (e.g., 512‑dim). Common choices include BERT/USE/SimCSE‑style encoders.

---

## 4. Key Hyperparameters & Tips

| Parameter     | Typical Range | Meaning & Tips                                                                                |
| ------------- | ------------- | --------------------------------------------------------------------------------------------- |
| `max_K`       | 64–256        | Upper bound of NN retrieval. Larger = stabler but more costly; balance with data size & VRAM. |
| `kde_K`       | 16–64         | #neighbors in KDE. Smaller = more sensitive; larger = smoother. Usually `kde_K ≤ max_K`.      |
| `sigma`       | 0.5–2.0       | KDE bandwidth. Too small ⇒ noisy; too large ⇒ oversmoothing.                                  |
| `alpha`       | 0.3–0.7       | Trade‑off between representativeness (density) and coverage (diversity).                      |
| `C`           | 0.01–1.0      | Selection ratio/threshold or regularization strength depending on implementation.             |
| `sample_size` | 500–5000      | Candidate pool size per selection step; heavily impacts speed & quality.                      |
| `model_name`  | —             | Path/name of the sentence encoder (local BERT/USE/SimCSE, etc.).                              |
| `cache_dir`   | —             | Cache directory for intermediate artifacts and resume‑from‑cache.                             |

---

## 5. Component Config (`components.yaml`)

**Path:** `DataFlex/src/dataflex/configs/components.yaml`

**Preset example**

```yaml
tsds:
  name: tsds
  params:
    max_K: 128
    kde_K: 64
    sigma: 0.8
    alpha: 0.5
    C: 10.0
    model_name: "/home/lianghao/yry/TSDS/bert_chinese"
    cache_dir: ../dataflex_saves/tsds_output
```

---

## 6. Dynamic Training Config (LoRA + TSDS)

**Example file:** `DataFlex/examples/train_lora/selectors/tsds.yaml`

```yaml
### model
model_name_or_path: /home/lianghao/yry/LLaMA-Factory/Qwen2.5-0.5B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 8
# deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: alpaca_en_demo
template: qwen
cutoff_len: 4096
# max_samples: 100000000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 0
# disable_shuffling: true
seed: 42

### output
output_dir: ../dataflex_saves/qwen/tsds
logging_steps: 10
save_steps: 100
plot_loss: true
save_only_model: false
overwrite_output_dir: true

### swanlab
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]
# use_swanlab: true
# swanlab_project: medical_dynamic_sft
# swanlab_run_name: qwen2_5_3b_lora_medical_50k_baseline
# swanlab_workspace: word2li
# swanlab_api_key: <YOUR_KEY>
# swanlab_lark_webhook_url: <YOUR_WEBHOOK>
# swanlab_lark_secret: <YOUR_SECRET>

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: false

### Dataflex args
train_type: dynamic_select   # trainer type:
                             # "dynamic_select" | "dynamic_mix" | "dynamic_weight" | "static"
components_cfg_file: src/dataflex/configs/components.yaml
component_name: tsds  # must match the name in components_cfg_file
warmup_step: 400
update_step: 500
update_times: 2
# eval_dataset: alpaca_zh_demo
eval_dataset: alpaca_zh_demo
```

**Notes:**

* `component_name: tsds` enables the TSDS component.
* `warmup_step / update_step / update_times` decide **when** and **how often** to re‑select the training subset; total steps ≈ `warmup_step + update_step × update_times`.
* `eval_dataset` provides the **target distribution** reference for similarity/representativeness scoring.

---

## 7. Run Training

```bash
FORCE_TORCHRUN=0 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/tsds.yaml
```

**Note:** the above example runs without distributed launch.

During training, TSDS is triggered at scheduled steps: encode training samples → NN search / KDE → combine with diversity → select the next training subset.

---

## 8. Merge & Export the Model

Same as the Less Selector pipeline.

**Config file:** `DataFlex/examples/merge_lora/llama3_lora_sft.yaml`

```yaml
model_name_or_path: <base model path>
adapter_name_or_path: <finetuned adapter path>
template: qwen
trust_remote_code: true

export_dir: ../dataflex_saves/Qwen2.5-0.5B_lora_sft
export_size: 5
export_device: cpu
export_legacy_format: false
```

Run the export command (inside the LLaMA‑Factory directory):

```bash
llamafactory-cli export llama3_lora_sft.yaml
```

---

## 9. Evaluation & Comparison

We recommend using the [DataFlow](https://github.com/OpenDCAI/DataFlow) QA evaluation pipeline to compare **TSDS** against **Less** and **random sampling**. 


