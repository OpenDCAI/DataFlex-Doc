---
title: Offline-Tsds 数据选择器
createTime: 2025/11/01 21:35:45
permalink: /zh/guide/vkqfowej/
icon: tdesign:cat

---


# Offline TSDS Selector 使用介绍

本文档介绍如何在 **DataFlex** 框架中使用 **Offline TSDS Selector** Data Selection for Task-Specific Model Finetuning实现训练数据的**动态选择**，以在监督微调（SFT）中兼顾**密度代表性**与**多样性**，提升泛化效果。

---

## 1. 方法概述

**TSDS** 的核心思想是：

* 先将**已分词(tokenized)**的样本进一步编码为**句向量**（例如 512 维）。
* 在嵌入空间中进行**近邻搜索 & 密度估计（KDE）**，得到每个样本的“代表性分数”。
* 同时考虑**拓扑多样性**（避免只挑“挤在一起”的样本），在密度与多样性之间用系数 `alpha` 做权衡。

> 直观理解：密度高 = 更“典型/代表”的数据，
> 多样性高 = 覆盖面更广、减少信息冗余。

### 评分构成

设样本的句向量为 $e_i$，其 $K$ 个近邻集合为 $\mathcal{N}_K(i)$。

1. **核密度估计（KDE）**：
$$
\text{density}(i)
= \frac{1}{K} \sum_{j\in \mathcal{N}_K(i)}
\exp\!\left(-\frac{\lVert e_i - e_j \rVert^2}{2\sigma^2}\right)
$$

2. **多样性（简单实现可用去冗余惩罚/边际增益）**：
$$
\text{diversity}(i)\ \propto\
\min_{j\in S} \lVert e_i - e_j \rVert,\quad
S=\text{已选集合}
$$

3. **综合评分**：
$$
\text{score}(i)
= \alpha\, \text{density}(i)
+ (1-\alpha)\, \text{diversity}(i)
$$

> 实际实现中，`kde_K`（用于密度估计的近邻数）与 `max_K`（总检索近邻上限）可不同；`C` 可作为筛选比例/阈值等控制量。

---

## 2. 环境与依赖

```bash
# DataFlex（建议源码安装）
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .

# 训练与推理的常用依赖
pip install llamafactory==0.9.3

# TSDS 额外依赖（向量检索与进度条等）
pip install faiss-cpu vllm sentence-transformer
```

---

## 3. offline 数据选择

在DataFlex\src\dataflex\offline_selector\offline_tsds_selector.py文件中修改训练集、编码模型和参数
```python
if __name__ == "__main__":
    tsds = offline_tsds_Selector(
        candidate_path="OpenDCAI/DataFlex-selector-openhermes-10w",#训练集
        query_path="OpenDCAI/DataFlex-selector-openhermes-10w",#验证集

        # If you want to use vllm,please add "vllm:" before model's name
        # Otherwise it automatically use sentence-transfromer
        embed_model="vllm:Qwen/Qwen3-Embedding-0.6B",#编码模型
        batch_size=32,
        save_probs_path="tsds_probs.npy",
        max_K=5000,
        kde_K=1000,
        sigma=0.75,
        alpha=0.6,
        C=5.0
    )
    tsds.selector()
       
```

> **注意**：此处的 `model_name` 用于将**tokenized**后的文本进一步编码为**句向量**（例如 512 维），支持vllm和sentence-transformer 推理。

**最终保存为每个训练样本的采样概率**

---

## 4. 关键超参数与建议

| 参数            | 典型范围     | 含义与建议                                     |
| ------------- | -------- | ----------------------------------------- |
| `max_K`       | 64-2000  | 近邻检索数量上限，越大越稳但开销更高；建议与数据规模/显存权衡           |
| `kde_K`       | 16–10000 | 用于密度估计的邻居数，越小更敏感、越大更平滑；通常 `kde_K ≤ max_K` |
| `sigma`       | 0.5–2.0  | KDE 的核宽度，过小噪声大，过大易过平滑                     |
| `alpha`       | 0.3–0.7  | 密度 vs 多样性的权衡系数，靠 1 偏重代表性，靠 0 偏重覆盖度        |
| `C`           | 0.01–1.0 | 用作筛选比例/阈值/正则系数等控制量；与实现细节相关                |
| `sample_size` | 500–5000 | 每次候选评估的样本数上限；大幅影响速度与效果                    |
| `model_name`  | —        | 句向量编码模型路径或名称（如本地embedding模型）             |
| `cache_dir`   | —        | 中间结果缓存路径，便于断点续跑                           |

---

## 5. 组件配置（components.yaml）

**路径：** `DataFlex/src/dataflex/configs/components.yaml`

**预设参数**

```yaml
tsds:
    name: tsds
    params:
      probs_path: ./src/dataflex/offline_selector/tsds_probs.npy 
      #默认离线数据选择所在位置的tsds_probs.npy文件
      cache_dir: ../dataflex_saves/tsds_output
```

---

## 6. 动态训练配置（LoRA + TSDS）

**示例文件：** `DataFlex/examples/train_lora/selectors/tsds.yaml`

```yaml
### model
model_name_or_path: 
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
dataset: #训练集
template: qwen （训练模型类型：qwen、llama...）
cutoff_len: 4096
# max_samples: 100000000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 0
# disable_shuffling: true
seed: 42

### output
output_dir: ../dataflex_saves
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
# swanlab_api_key: AnLWTMijcbd4cyEfundi3
# swanlab_lark_webhook_url: https://open.feishu.cn/open-apis/bot/v2/hook/ff10a391-4e51-4481-97ff-965760cae2a1
# swanlab_lark_secret: cySzwTbCJh08349FGAhBSf

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
train_type: dynamic_select   # 选择训练器类型。可选值包括：
                          # "dynamic_select" - 动态选择训练器
                          # "dynamic_mix" - 动态混合训练器
                          # "dynamic_weight" - 动态加权训练器
                          # "static" - 默认静态训练器
components_cfg_file: src/dataflex/configs/components.yaml
component_name: tsds  # 选择组件名称，对应 components_cfg_file 中定义的组件
warmup_step: 400
update_step: 500
update_times: 2
# eval_dataset: alpaca_zh_demo

```

**参数说明：**

* `component_name: tsds`：启用 TSDS 组件。
* `warmup_step / update_step / update_times`：决定**何时**与**多久**进行一次动态选择；总步数 ≈ `warmup_step + update_step × update_times`。
*  总batch_size=device_number x per_device_train_batch_size x gradient_accumulation_steps

---

## 7. 运行训练

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/tsds.yaml
```
**采用分布式**

训练过程中会在设定的步数触发 TSDS 动态选择：根据离线选择的样本采样概率，选出下一阶段训练子集。

---

## 8. 模型合并与导出

与 Less Selector 流程一致：

**配置文件：** `DataFlex/examples/merge_lora/llama3_lora_sft.yaml`

```yaml
model_name_or_path: 原模型地址
adapter_name_or_path: 微调后adpter地址
template: qwen
trust_remote_code: true

export_dir: ../dataflex_saves
export_size: 5
export_device: cpu
export_legacy_format: false
```

导出命令：
在llamafactory文件夹中运行
```bash
llamafactory-cli export llama3_lora_sft.yaml
```

---

## 9. 评估与对比

建议使用 [DataFlow](https://github.com/OpenDCAI/DataFlow) 的模型 QA 评估流水线，对 **TSDS** 与 **Less**、**随机采样** 等策略进行并列评测