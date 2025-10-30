---
title: Less Selector
createTime: 2025/10/30 16:46:08
permalink: /zh/guide/qlrrfg9b/
icon: solar:add-circle-outline
---
# Less Selector 使用介绍

> 本文档介绍如何在 **DataFlex** 框架中使用 **Less Selector** 实现训练数据的动态选择，从而提升监督微调（SFT）效果。
> 该方法源于[**Less: Sample Selection via Influence Functions** (ICML 2024)](https://dl.acm.org/doi/10.5555/3692070.3694291)。

---

## 📘 1. 方法概述

**Less Selector** 的核心思想是：
基于Adam优化器的**样本影响函数（Influence Function）**，通过梯度方向的相似性来度量训练样本与验证样本的相关性。在 SFT 过程中动态选择训练样本，以提升模型的泛化性能。

### 数学定义

$$
\mathrm{Inf}_{\mathrm{less}}(z, z') 
\triangleq 
\sum_{i=1}^{N} 
\bar{\eta}_i 
\cos \big( 
\nabla \ell(z'; \theta_i), 
\Gamma(z, \theta_i) 
\big)
$$

**参数说明：**
* `z, z'`: 输入样本或数据点，z来自验证集，z'来自训练集。
* `N`: 训练时，数据动态选择的次数。
* `\overline{\eta}_i`: 第 `i` 次选择的有效学习率。
* `\ell(z'; \theta_i)`: 样本 `z'` 在参数 `\theta_i` 下的损失，用于计算该步的梯度信号。
* `\nabla \ell(z'; \theta_i)`: 损失对参数的梯度，表示样本 `z'` 在步 `i` 产生的更新方向。
* `\Gamma(z, \theta_i)`: 样本 `z` 的影响向量，表示在步 `i` 下样本 `z` 对参数更新方向的作用。
* `\cos(\bullet, \bullet)`: 余弦相似度，衡量两个向量方向一致性的度量，用于对齐上述两方向。

---

## ⚙️ 2. 实现步骤

### 步骤一：环境安装

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory
```

---

### 步骤二：Less Selector 参数配置

**配置文件路径：**
```
DataFlex/src/dataflex/configs/components.yaml
```

**示例配置：**
```yaml
less:
  name: less
  params:
    cache_dir: ../dataflex_saves/less_output
    gradient_type: adam
    proj_dim: 4096
    seed: 123
```

**参数说明：**
* `gradient_type`: 使用的梯度下降类型，默认`adam`。
* `proj_dim`: 随机投影维度，（如 `4096` 或 `8192`），用于降低计算成本，详见[Less](https://dl.acm.org/doi/10.5555/3692070.3694291)中“4.1 Step 2: Projecting the gradients”。
* `cache_dir`: 保存中间结果的缓存路径。
* `seed`: 随机种子，确保可复现性。

---

### 步骤三：动态训练配置

**配置文件路径：**

```
DataFlex/examples/train_lora/selectors/less.yaml
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
deepspeed: examples/deepspeed/ds_z3_config.json  

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 0
seed: 42

### output
output_dir: ../dataflex_saves/Llama-3.1-8B/less
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

### dynamic_train
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: less
warmup_step: 10
update_step: 10
update_times: 2

eval_dataset: alpaca_zh_demo
```

**参数说明：**
* `model_name_or_path`: 监督微调训练模型的名称或路径。
* `dataset`: 训练数据集。
* `output_dir`: 动态微调结果（LoRA 适配器）的输出路径。
* `warmup_step`: 训练初期第一次训练数据选择前，进行warmup的步数。
* `update_step`: 每次训练数据动态选择的步数。
* `update_times`: 数据动态选择的总次数。
* `eval_dataset`: 验证数据集。

> dataset和eval_dataset可选`DataFlow/qmy/DataFlex/data/dataset_info.json`中数据，或本地路径下sharegpt或alpaca格式的json数据。注意该方法的情形下，训练集规模会较大影响计算成本。
> 总步数 = `warmup_step + update_step × update_times`

---

### 步骤四：运行训练


```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/less.yaml

```
训练过程会自动完成动态数据选择与模型更新。

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

执行导出命令：

```bash
llamafactory-cli export llama3_lora_sft.yaml
```

合并后的模型将保存在如下文件夹：

```
/dataflex_saves/Llama-3.1-8B_lora_sft
```
---

## 🧩3. 模型评估

推荐使用[DataFlow](https://github.com/OpenDCAI/DataFlow)的[模型QA能力评估流水线](https://opendcai.github.io/DataFlow-Doc/zh/guide/2k5wjgls/)对生成后的模型进行系统性评估。

