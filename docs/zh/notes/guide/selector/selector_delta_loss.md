---
title: Delta Loss 数据选择器
createTime: 2025/12/27 00:44:11
permalink: /zh/guide/delta-loss/
icon: carbon:select-window
---
# Delta Loss Selector 使用介绍

本文档介绍如何在 **DataFlex** 框架中使用 **Delta Loss Selector**。该方法以“初始损失 - 当前损失”作为样本有效性的信号，结合滑动窗口策略进行动态采样，从而在训练过程中逐步移动关注区间。

---

## 1. 方法概述

**Delta Loss Selector** 的核心流程：

1. 首次选择时计算并缓存 **初始损失**，并随机采样作为 warmup。
2. 后续步骤计算当前损失，得到 $\Delta l_i = l_i^{(init)} - l_i^{(current)}$。
3. 对 $\Delta l_i$ 降序排序，并按训练进度计算滑动窗口位置。
4. 对窗口内样本赋较高采样概率，窗口外样本保留较低基准概率。

**滑动窗口位置：**

设当前更新次数为 $t$，总更新次数为 $T$，窗口大小为 $s$（比例），
窗口起点通过 Sigmoid 调度获得：

$$
u = \sigma\left(\frac{t}{T}\right), \quad
\text{start} = u \cdot (N - sN), \quad
\text{end} = \text{start} + sN
$$

当 $\Delta l_i < 0$ 时，窗口右端会被截断，避免选择损失变差的样本。


## 2. 实现步骤

### 步骤一：环境安装

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory
```

---

### 步骤二：Delta Loss Selector 参数配置

**配置文件路径：**
```
DataFlex/src/dataflex/configs/components.yaml
```

**示例配置：**
```yaml
delta_loss:
  name: delta_loss
  params:
    cache_dir: ../dataflex_saves/delta_loss_output
    window_size: 0.2
```

**参数说明：**
* `cache_dir`: 选择结果缓存路径，首次选择会保存 initial loss。
* `window_size`: 滑动窗口大小（比例），决定每轮重点采样的区间宽度。

---

### 步骤三：动态训练配置

**配置文件路径：**
```
DataFlex/examples/train_lora/selectors/delta_loss.yaml
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

`update_times` 表示每个 Flex epoch 内的动态选择次数。Delta Loss 需要 `update_times >= 2`：第一次选择建立 initial loss，后续选择根据 delta loss 更新样本。多 epoch 训练请设置 `num_train_epochs: N` 且保持 `train_step: 0`。

---

### 步骤四：运行训练

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/delta_loss.yaml
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
adapter_name_or_path: ../dataflex_saves/Llama-3.1-8B/delta_loss
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
