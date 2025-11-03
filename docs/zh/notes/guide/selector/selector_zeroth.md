# 零阶选择器介绍
本文档介绍如何在 **DataFlex** 框架中使用 **Zeroth Selector** 实现训练数据的动态选择，从而提升监督微调（SFT）效果。该方法为原创选择方法，利用对模型进行同向扰动进行差分估计，进而得到模型的零阶梯度来计算数据有效分数。

---

## 1. 方法概述

**Zeroth Selector** 的核心思想是：
基于**SGD**版本的**样本影响函数（Influence Function）**，通过零阶梯度方向的相似性来度量训练样本与验证样本的相关性。值得注意的是，对于训练数据和验证数据，采用的扰动噪声是使用相同的随机数种子生成的，所以目前版本的选择算法可以直接使用**差分（投影梯度）** 的乘积代表梯度内积相似度。优势：1. 避免直接使用**梯度样本影响函数** 只能逐样本进行计算的无法数据并行的问题；2. 不需要反传，节省时间和显存开销。

### 数学定义
随机采样$\xi\sim\mathbf{N}(0,I)$，对模型参数$\theta$进行扰动，
$$
\mathrm{Inf}_{\mathrm{zeroth}}(z,z'):=(\dfrac{f(\theta+\epsilon\xi;z)-f(\theta-\epsilon\xi;z)}{2\epsilon})\cdot(\dfrac{f(\theta+\epsilon\xi;z')-f(\theta-\epsilon\xi;z')}{2\epsilon})
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

### 步骤二：Zeroth Selector 参数配置

**配置文件路径：**
```
DataFlex/src/dataflex/configs/components.yaml
```

**示例配置：**
```yaml
less:
  name: zeroth
  params:
    cache_dir: ../dataflex_saves/zeroth_output
    seed: 42
```

**参数说明：**
* `cache_dir`: 保存中间结果的缓存路径，即中间差分值。
* `seed`: 可选，随机种子，用作采样噪声的生成器种子。

---

### 步骤三：动态训练配置

**配置文件路径：**

```
DataFlex/examples/train_lora/selectors/zeroth.yaml
```

**示例配置：**

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 8
#deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: alpaca_en_demo
#dataset: flan_v2,cot_data,dolly_data,oasst1_data
#eval_dataset: mmlu_eval
template: qwen
cutoff_len: 4096
# max_samples: 100000000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 0
# disable_shuffling: true
seed: 42

### output
output_dir: /data1/xlyang/Flex/saves/zeroth/
logging_steps: 10
save_steps: 100
plot_loss: true
save_only_model: false
overwrite_output_dir: true

### swanlab
report_to: none  # choices: [none, wandb, tensorboard, swanlab, 

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: false
fp16: true
ddp_timeout: 180000000

### dynamic_train
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: zeroth
warmup_step: 4
update_step: 3
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

dataset和eval_dataset可选`DataFlex/data/dataset_info.json`中数据，或本地路径下sharegpt或alpaca格式的json数据。注意该方法的情形下，训练集规模会较大影响计算成本。

总步数 = `warmup_step + update_step × update_times`。

---

### 步骤四：运行训练


```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/zeroth.yaml

```
训练过程会自动完成动态数据选择与模型更新。

---


## 3. 模型评估

推荐使用[DataFlow](https://github.com/OpenDCAI/DataFlow)的[模型QA能力评估流水线](https://opendcai.github.io/DataFlow-Doc/zh/guide/2k5wjgls/)对生成后的模型进行系统性评估。

