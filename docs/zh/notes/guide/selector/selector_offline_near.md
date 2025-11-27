---
title: Offline-Near数据选择器
createTime: 2025/11/26 23:42:41
permalink: /zh/guide/acgesu99/
icon: flowbite:fish-alt-outline
---
# Offline NEAR Selector 使用介绍

本文档介绍如何在 **DataFlex** 框架中使用 **Offline NEAR Selector** 实现训练数据的**动态选择**，以在监督微调（SFT）中聚焦于与目标集的相似度，进行邻近选择。

---

## 1. 方法概述

**NEAR** 的核心思想是：

* 先将**已分词(tokenized)**的样本进一步编码为**句向量**（例如 512 维）。
* 在嵌入空间中进行**近邻搜索 **，得到每个样本与目标集的“样本相似度”。


> 直观理解：选择与目标集最接近的训练数据，以最优化训练目标。


---

## 2. 环境与依赖

```bash
# DataFlex（建议源码安装）
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .

# 训练与推理的常用依赖
pip install llamafactory==0.9.3

# NEAR 额外依赖（向量检索与进度条等）
pip install faiss-cpu vllm sentence-transformer
```

---

## 3. offline 数据选择

在DataFlex\src\dataflex\offline_selector\offline_near_selector.py文件中修改训练集、编码模型和参数
```python
if __name__ == "__main__":
    near = offline_near_Selector(
        candidate_path="OpenDCAI/DataFlex-selector-openhermes-10w", # split = train
        query_path="OpenDCAI/DataFlex-selector-openhermes-10w", # split = vaildation
        
        # If you want to use vllm,please add "vllm:" before model's name
        # Otherwise it automatically use sentence-transfromer
        embed_model="vllm:Qwen/Qwen3-Embedding-0.6B",
        batch_size=32,
        save_indices_path="top_indices.npy",
        max_K=1000,
        
    )
    near.selector()
       
```

> **注意**：此处的 `model_name` 用于将**tokenized**后的文本进一步编码为**句向量**（例如 512 维），支持vllm和sentence-transformer 推理。

**最终保存为每个query的max_K个最邻近训练数据的索引矩阵 （ N ,max_K ）**

---

## 4. 关键超参数与建议

| 参数            | 典型范围     | 含义与建议                                     |
| ------------- | -------- | ----------------------------------------- |
| `max_K`       | 64–2000   | 近邻检索数量上限，越大越稳但开销更高；建议与数据规模/显存权衡           |                   |
| `model_name`  | —        | 句向量编码模型路径或名称（如本地embeddingm模型）             |
| `cache_dir`   | —        | 中间结果缓存路径，便于断点续跑                           |

---

## 5. 组件配置（components.yaml）

**路径：** `DataFlex/src/dataflex/configs/components.yaml`

**预设参数**

```yaml
near:
    name: near
    params:
      indices_path: ./src/dataflex/offline_selector/top_indices.npy
      cache_dir: ../dataflex_saves/near_output
  
```

---

## 6. 动态训练配置（LoRA + NEAR）

**示例文件：** `DataFlex/examples/train_lora/selectors/near.yaml`

```yaml
### model
model_name_or_path: #模型地址
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
component_name: near  # 选择组件名称，对应 components_cfg_file 中定义的组件
warmup_step: 400
update_step: 500
update_times: 2
# eval_dataset: alpaca_zh_demo


```

**参数说明：**

* `component_name: near`：启用 NEAR 组件。
* `warmup_step / update_step / update_times`：决定**何时**与**多久**进行一次动态选择；总步数 ≈ `warmup_step + update_step × update_times`。
*  总batch_size=device_number x per_device_train_batch_size x gradient_accumulation_steps


---

## 7. 运行训练

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/near.yaml
```
**采用分布式**

训练过程中会在设定的步数触发 NEAR 动态选择：根据离线选择的样本索引，选出下一阶段训练子集。

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

建议使用 [DataFlow](https://github.com/OpenDCAI/DataFlow) 的模型 QA 评估流水线，对 **NEAR** 与 **Less**、**随机采样** 等策略进行并列评测