---
title: NICE Data Selector
createTime: 2025/12/17 12:00:08
permalink: /en/guide/nice/
icon: carbon:select-02
---
# NICE Selector Usage Guide

This document introduces how to use the **NICE Selector** for dynamic data selection in the **DataFlex** framework. The method constructs gradient similarity between the training set and the validation set: the training set uses SFT loss gradients, while the validation set uses policy gradients based on a reward model. After random projection, similarities are computed to select training samples that are most aligned with the target samples. This method is based on  
[**NICE Data Selection for Instruction Tuning in LLMs with Non-differentiable Evaluation Metric** (ICML 2025)](https://icml.cc/virtual/2025/poster/46560).

---

## 1. Method Overview

The core workflow of **NICE Selector**:

1. **Data normalization**: Automatically supports formats such as Alpaca and ShareGPT.
2. **Training-set gradients**: Compute gradients for each training sample and project them using TRAK.
3. **Reward-set gradients**: Perform Monte Carlo sampling on validation data, generate responses, score them using a reward model (local vLLM or remote API), compute policy gradients toward the reward direction, and project them.
4. **Similarity-based selection**: Align and normalize projected gradients, rank training samples by their average similarity to validation samples, and select the top-k samples for the current training round.


## 2. Implementation Steps

### Step 1: Environment Setup

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory
````

---

### Step 2: NICE Selector Configuration

**Configuration file path:**

```
DataFlex/src/dataflex/configs/components.yaml
```

**Example configuration:**

```yaml
nice:
  name: nice
  params:
    cache_dir: ../dataflex_saves/nice_output
    gradient_type: adam
    proj_dim: 4096
    seed: 123
    save_interval: 16
    reward_model_backend: local_vllm   # choices: [local_vllm, api]
    reward_backend_params:
      local_vllm:
        hf_model_name_or_path: meta-llama/Llama-3.1-8B
        vllm_tensor_parallel_size: 1
        vllm_temperature: 0.0
        vllm_top_p: 0.9
        vllm_max_tokens: 512
        vllm_top_k: 40
        vllm_seed: 42
        vllm_max_model_len: null
        vllm_gpu_memory_utilization: 0.9
      api:
        api_url: https://api.openai.com/v1/chat/completions
        api_key: DF_API_KEY
        model_name: gpt-4o
        temperature: 0.0    
    mc_samples: 4
    max_new_tokens: 512
    generation_temperature: 0.7
    max_prompt_length: 4096
```

**Parameter description:**

* `cache_dir`: Path to cache gradient projections and selection results; supports resuming from checkpoints.
* `gradient_type`: `adam` (with first- and second-moment normalization) or `sgd`.
* `proj_dim`: Random projection dimension, controlling the cost/accuracy trade-off of similarity computation.
* `reward_model_backend`: Reward model backend; `local_vllm` uses local vLLM inference, `api` uses an HTTP service.
* `reward_backend_params`: Backend-specific parameters.
* `mc_samples`: Number of Monte Carlo generations per reward sample, used to stabilize policy gradient estimation.
* `max_new_tokens` / `generation_temperature` / `max_prompt_length`: Generation length and sampling strategy for the policy model.

---

### Step 3: Dynamic Training Configuration

**Configuration file path:**

```
DataFlex/examples/train_lora/selectors/nice.yaml
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
output_dir: ../dataflex_saves/nice_output
logging_steps: 10
save_steps: 100
plot_loss: true
save_only_model: false
overwrite_output_dir: true

### swanlab
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]
# use_swanlab: true
# swanlab_project: dynamic_nice_sft
# swanlab_run_name: name
# swanlab_workspace: your_workspace
# swanlab_api_key: xxxxxxx

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### dynamic_train
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: nice
warmup_step: 10
update_step: 10
update_times: 2

eval_dataset: alpaca_zh_demo
per_device_eval_batch_size: 1
metric_for_best_model: eval_loss
greater_is_better: false
load_best_model_at_end: true
eval_strategy: steps    # choices: [no, steps, epoch]
eval_steps: 10
early_stopping_steps: 3
early_stopping_min_delta: 0.01
```

**Parameter description:**

* `component_name`: Must match the `nice` component in `components.yaml`, determining reward backend and projection dimensions.
* `warmup_step` / `update_step` / `update_times`: Control the dynamic selection schedule; total steps = `warmup_step + update_step × update_times`.
* `eval_dataset`: Validation set (Alpaca/ShareGPT style); reward model is used for scoring during generation.
* `output_dir`: Path to save LoRA adapters and caches.

---

### Step 4: Run Training

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/nice.yaml
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
adapter_name_or_path: ../dataflex_saves/nice_output
template: llama3
trust_remote_code: true

export_dir: ../dataflex_saves/Llama-3.1-8B_nice_lora_sft
export_size: 5
export_device: cpu  # choices: [cpu, auto]
export_legacy_format: false
```

**Parameter description:**

* `adapter_name_or_path`: Path to the LoRA adapters obtained from NICE dynamic selection training.
* `export_dir`: Output directory for the merged full model.

Run the merge and export command:

```bash
llamafactory-cli export llama3_lora_sft.yaml
```

The merged model will be saved to:

```
/dataflex_saves/Llama-3.1-8B_nice_lora_sft
```


## 3. Model Evaluation

It is recommended to use the [DataFlow](https://github.com/OpenDCAI/DataFlow)
[Model QA Evaluation Pipeline](https://opendcai.github.io/DataFlow-Doc/en/guide/2k5wjgls/) to systematically evaluate the generated model, and to inspect the scoring logs in `cache_dir` to analyze the reward model’s sensitivity to different samples.

