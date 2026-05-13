---
title: Delta Loss Selector
createTime: 2025/12/27 00:44:11
permalink: /en/guide/delta-loss/
icon: carbon:select-window
---
# Delta Loss Selector Guide

This document explains how to use the **Delta Loss Selector** in **DataFlex**. It tracks loss reduction relative to an initial baseline and samples from a sliding window over the ranked delta-loss list.

---

## 1. Method Overview

**Delta Loss Selector** workflow:

1. On the first selection, compute and cache **initial losses**, then return a random warmup batch.
2. On later steps, compute current losses and define $\Delta l_i = l_i^{(init)} - l_i^{(current)}$.
3. Sort samples by $\Delta l_i$ in descending order and compute a sliding-window position based on training progress.
4. Assign high sampling probability inside the window and a small base probability outside.

**Sliding window schedule:**

Let update index be $t$ out of $T$, and window ratio be $s$:

$$
u = \sigma\left(\frac{t}{T}\right), \quad
\text{start} = u \cdot (N - sN), \quad
\text{end} = \text{start} + sN
$$

If $\Delta l_i < 0$, the window end is truncated to avoid prioritizing samples that got worse.


## 2. Implementation Steps

### Step 1: Environment Setup

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory
```

---

### Step 2: Delta Loss Selector Configuration

**Configuration file path:**
```
DataFlex/src/dataflex/configs/components.yaml
```

**Example configuration:**
```yaml
delta_loss:
  name: delta_loss
  params:
    cache_dir: ../dataflex_saves/delta_loss_output
    window_size: 0.2
```

**Parameter Description:**
* `cache_dir`: Cache directory; the first step stores initial losses here.
* `window_size`: Sliding window ratio for focused sampling.

---

### Step 3: Dynamic Training Configuration

**Configuration file path:**
```
DataFlex/examples/train_lora/selectors/delta_loss.yaml
```

**Example configuration:**
```yaml
### model
model_name_or_path: meta-llama/Llama-3.1-8B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 8

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 0
seed: 42

### output
output_dir: ../dataflex_saves/Llama-3.1-8B/delta_loss
logging_steps: 10
save_steps: 100
plot_loss: true
save_only_model: false
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### Dataflex args
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: delta_loss
warmup_step: 10
update_step: 10
update_times: 2

eval_dataset: alpaca_zh_demo
```

`update_times` is the number of dynamic selections per Flex epoch. Delta Loss requires `update_times >= 2`: the first selection builds initial losses, and later selections use delta loss. For multi-epoch runs, set `num_train_epochs: N` and keep `train_step: 0`.

---

### Step 4: Run Training

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/delta_loss.yaml
```

---

### Step 5: Model Merge and Export

**Configuration file path:**

```
DataFlex/examples/merge_lora/llama3_lora_sft.yaml
```

**Example configuration:**

```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: ../dataflex_saves/Llama-3.1-8B/delta_loss
template: llama3
trust_remote_code: true

export_dir: ../dataflex_saves/Llama-3.1-8B_lora_sft
export_size: 5
export_device: cpu  # choices: [cpu, auto]
export_legacy_format: false
```

**Parameter Description:**

* `model_name_or_path`: Model name or path used for training.
* `adapter_name_or_path`: Output path of the LoRA adapter.
* `export_dir`: Directory for saving the merged result of the fine-tuned model and LoRA adapter.

Execute the export command:

```bash
llamafactory-cli export llama3_lora_sft.yaml
```

The merged model will be saved in:

```
/dataflex_saves/Llama-3.1-8B_lora_sft
```


## 3. Model Evaluation

It is recommended to use the [DataFlow](https://github.com/OpenDCAI/DataFlow) [Model QA Evaluation Pipeline](https://opendcai.github.io/DataFlow-Doc/zh/guide/2k5wjgls/) for systematic evaluation of the generated model.
