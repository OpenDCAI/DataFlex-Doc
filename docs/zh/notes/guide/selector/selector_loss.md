---
title: Loss 数据选择器
createTime: 2025/12/27 00:44:11
permalink: /zh/guide/loss/
icon: carbon:select-window
---
# Loss Selector 使用介绍

本文档介绍如何在 **DataFlex** 框架中使用 **Loss Selector**，基于样本损失分布进行动态数据选择。该方法会按分位数将样本划分为低/中/高损失区间，并对指定区间加权采样，以便在训练中更聚焦特定难度的样本。

---

## 1. 方法概述


**Loss Selector** 的核心思想是：
1. 在训练过程中计算样本对应的训练损失，并在多卡环境下通过样本索引（`idx`）将各进程结果对齐到完整数据集。

   * 当前实现使用 `batch_size=1`，因此模型返回的 loss 等价于逐样本损失。
2. 在主进程上收集并去重所有样本的有效损失值，使用分位数阈值将样本划分为 **low / medium / high** 三个损失区间。
3. 为所有有效样本赋予基础权重 1，并对指定关注区间（`focus`）内的样本施加放大权重 `focus_weight`。
4. 通过温度参数对权重分布进行平滑，并依据得到的概率分布进行随机采样；当有效样本数量不足以满足采样需求时，自动切换为放回采样。

**采样概率：**

设样本损失为 $l_i$，分段权重为 $w_i$，温度为 $T$： 

$$
 p_i = \frac{(w_i + \epsilon)^{1/T}}{\sum_j (w_j + \epsilon)^{1/T}} 
$$


## 2. 实现步骤

### 步骤一：环境安装

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory
```

---

### 步骤二：Loss Selector 参数配置

**配置文件路径：**
```
DataFlex/src/dataflex/configs/components.yaml
```

**示例配置：**
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

**参数说明：**
* `cache_dir`: 选择结果缓存路径（每个 step 会写入 `step_{id}.json`）。
* `focus`: 关注区间，可选 `low` / `medium` / `high`（默认 `high`）。
* `focus_weight`: 关注区间的权重倍数，越大越偏向该区间。
* `quantiles`: 低/中/高损失的分位数切分点，取值在 `[0, 1]`。
* `replacement`: 是否放回采样；若请求数大于有效样本量，会自动切换为放回采样。
* `temperature`: 温度系数，`>1` 更平滑，`<1` 更尖锐。

---

### 步骤三：动态训练配置

**配置文件路径：**
```
DataFlex/examples/train_lora/selectors/loss.yaml
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

`update_times` 表示每个 Flex epoch 内的动态选择次数。多 epoch 训练请设置 `num_train_epochs: N` 且保持 `train_step: 0`；若 `train_step > 0`，则固定总步数并覆盖 `num_train_epochs`。

---

### 步骤四：运行训练

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/loss.yaml
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
adapter_name_or_path: ../dataflex_saves/Llama-3.1-8B/less
template: llama3
trust_remote_code: true

export_dir: ../dataflex_saves/Llama-3.1-8B_lora_sft
export_size: 5
export_device: cpu  # choices: [cpu, auto]
export_legacy_format: false
```
**参数说明：**
* `model_name_or_path`: 训练模型的名称或路径。
* `adapter_name_or_path`: LoRA适配器输出路径。
* `export_dir`: 监督微调后的模型，训练模型与LoRA适配器的合并结果。

执行合并导出命令：

```bash
llamafactory-cli export llama3_lora_sft.yaml
```

合并后的模型将保存在如下文件夹：

```
/dataflex_saves/Llama-3.1-8B_lora_sft
```

## 3. 模型评估

推荐使用[DataFlow](https://github.com/OpenDCAI/DataFlow)的[模型QA能力评估流水线](https://opendcai.github.io/DataFlow-Doc/zh/guide/2k5wjgls/)对生成后的模型进行系统性评估。
