---
title: Nice 数据选择器
createTime: 2025/12/17 12:00:08
permalink: /zh/guide/nice/
icon: carbon:select-02
---
# NICE Selector 使用介绍

本文档介绍如何在 **DataFlex** 框架中使用 **NICE Selector** 进行动态数据选择。该方法在训练集与验证集之间构造梯度相似性：训练集使用SFT损失梯度，验证集则使用基于奖励模型的策略梯度；二者通过随机投影后计算相似度，挑选与目标样本最对齐的训练数据。该方法源于[**NICE Data Selection for Instruction Tuning in LLMs with Non-differentiable Evaluation Metric** (ICML 2025)](https://icml.cc/virtual/2025/poster/46560)。

---

## 1. 方法概述

**NICE Selector** 的核心流程：

1. **数据规范化**：自动兼容 Alpaca、ShareGPT 等格式。
2. **训练集梯度**：对训练集样本逐一计算梯度，并使用 TRAK 投影。
3. **奖励集梯度**：在验证数据上进行 Monte Carlo 采样生成，调用奖励模型（本地 vLLM 或远程 API）打分，使用策略梯度得到奖励方向的梯度并投影。
4. **相似度选择**：对齐并归一化投影梯度，按训练集与验证集的平均相似度排序，选出 top-k 作为本轮训练数据。

## 2. 实现步骤

### 步骤一：环境安装

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory
```

---

### 步骤二：NICE Selector 参数配置

**配置文件路径：**
```
DataFlex/src/dataflex/configs/components.yaml
```

**示例配置：**
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

**参数说明：**
* `cache_dir`: 缓存梯度投影与选择结果的路径，支持断点续传。
* `gradient_type`: `adam`（带一二阶矩归一化）或 `sgd`。
* `proj_dim`: 随机投影维度，决定梯度相似度计算的成本/精度。
* `reward_model_backend`: 奖励模型后端，如果`local_vllm` 使用本地 vLLM 推理，如果`api` 走 HTTP 服务。
* `reward_backend_params`: 不同后端的专属参数。
* `mc_samples`: 每个奖励样本的 Monte Carlo 生成次数，用于稳定策略梯度估计。
* `max_new_tokens` / `generation_temperature` / `max_prompt_length`: 策略模型生成回答的长度与采样策略。

---

### 步骤三：动态训练配置

**配置文件路径：**

```
DataFlex/examples/train_lora/selectors/nice.yaml
```

**示例配置：**

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
# swanlab_workspace: ypur_workspace
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

**参数说明：**
* `component_name`: 与 `components.yaml` 中的 `nice` 组件保持一致，决定奖励后端与投影维度等设置。
* `warmup_step` / `update_step` / `update_times`: 决定动态选择的触发节奏；总步数 = `warmup_step + update_step × update_times`。
* `eval_dataset`: 验证集，可以是 Alpaca/ShareGPT 样式，生成时会调用奖励模型打分。
* `output_dir`: LoRA 适配器与缓存保存路径。

---

### 步骤四：运行训练

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/nice.yaml
```

---

### 步骤五：模型合并与导出

**配置文件路径：**

```
DataFlex/examples/merge_lora/llama3_lora_sft.yaml
```

**示例配置：**

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
**参数说明：**
* `adapter_name_or_path`: NICE 动态选择训练得到的 LoRA 适配器路径。
* `export_dir`: 合并后完整模型的输出目录。

执行合并导出命令：

```bash
llamafactory-cli export llama3_lora_sft.yaml
```

合并后的模型将保存在：

```
/dataflex_saves/Llama-3.1-8B_nice_lora_sft
```

## 3. 模型评估

推荐使用 [DataFlow](https://github.com/OpenDCAI/DataFlow) 的 [模型 QA 能力评估流水线](https://opendcai.github.io/DataFlow-Doc/zh/guide/2k5wjgls/) 对生成后的模型进行系统评估，并结合 `cache_dir` 中的打分日志观察奖励模型对不同样本的敏感度。
