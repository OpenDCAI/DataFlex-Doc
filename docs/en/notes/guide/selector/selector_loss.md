---
title: Loss Selector
createTime: 2025/12/27 00:44:11
permalink: /en/guide/loss/
icon: carbon:select-window
---
# Loss Selector Guide

This document explains how to use the **Loss Selector** in **DataFlex**. The selector computes per-sample training loss, splits the distribution into low/medium/high bands using quantiles, and then samples with higher weight on a chosen band.

---

## 1. Method Overview

**Core idea of the Loss Selector:**

1. During training, compute the training loss for each sample. In a multi-GPU setting, results from different processes are aligned to the full dataset using the sample index (`idx`).

   * In the current implementation, `batch_size = 1`, so the loss returned by the model corresponds to a per-sample loss.
2. On the main process, collect and deduplicate all valid sample losses, and partition samples into **low / medium / high** loss regions using quantile thresholds.
3. Assign a base weight of 1 to all valid samples, and apply an amplified weight `focus_weight` to samples in the specified focus region (`focus`).
4. Smooth the weight distribution using a temperature parameter and perform random sampling according to the resulting probability distribution; when the number of valid samples is insufficient to meet the sampling requirement, automatically switch to sampling with replacement.

**Sampling probability:**

Let the loss of sample ($i$) be ($l_i$), the segment weight be ($w_i$), and the temperature be ($T$):

$$
 p_i = \frac{(w_i + \epsilon)^{1/T}}{\sum_j (w_j + \epsilon)^{1/T}} 
$$


## 2. Implementation Steps

### Step 1: Environment Setup

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory
```

---

### Step 2: Loss Selector Configuration

**Configuration file path:**
```
DataFlex/src/dataflex/configs/components.yaml
```

**Example configuration:**
```yaml
loss:
  name: loss
  params:
    cache_dir: ../dataflex_saves/loss_output
    focus: "medium"        # low | medium | high
    focus_weight: 5.0
    quantiles: [0.33, 0.66]
    replacement: false
    temperature: 1.0
```

**Parameter Description:**
* `cache_dir`: Cache directory for selection results (`step_{id}.json` per step).
* `focus`: Target band to up-weight (`low` / `medium` / `high`, default `high`).
* `focus_weight`: Weight multiplier for the focus band.
* `quantiles`: Split points for low/medium/high, values in `[0, 1]`.
* `replacement`: Sample with replacement or not; auto-switches to replacement if needed.
* `temperature`: Distribution sharpness; `>1` smooths, `<1` sharpens.

---

### Step 3: Dynamic Training Configuration

**Configuration file path:**
```
DataFlex/examples/train_lora/selectors/loss.yaml
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
output_dir: ../dataflex_saves/Llama-3.1-8B/loss
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
component_name: loss
warmup_step: 10
update_step: 10
update_times: 2

eval_dataset: alpaca_zh_demo
```

`update_times` is the number of dynamic selections per Flex epoch. For multi-epoch runs, set `num_train_epochs: N` and keep `train_step: 0`; if `train_step > 0`, it fixes total steps and overrides `num_train_epochs`.

---

### Step 4: Run Training

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/loss.yaml
```


### Step 5: Model Merge and Export

**Configuration file path:**

```
DataFlex/examples/merge_lora/llama3_lora_sft.yaml
```

**Example configuration:**

```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: ../dataflex_saves/Llama-3.1-8B/loss
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
