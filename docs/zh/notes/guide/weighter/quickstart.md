---
title: 快速开始
createTime: 2025/06/30 19:19:16
permalink: /zh/guide/weighter/quickstart/
icon: solar:bolt-outline
---

# 快速开始

启动命令类似于 [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory)。以下是使用损失加权器的示例：

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/weighters/loss.yaml
```

与普通的 LlamaFactory 不同，您的 `.yaml` 配置文件必须包含 **DataFlex 特定的参数**：

```yaml
### dynamic_train
train_type: dynamic_weight 
components_cfg_file: src/dataflex/configs/components.yaml
component_name: loss
warmup_step: 1
num_train_epochs: 1.0
train_step: 0 # 设为正数时固定总步数，并覆盖 num_train_epochs
```

多 epoch 训练只需设置 `num_train_epochs: N` 并保持 `train_step: 0`。`warmup_step` 是全局 step 阈值，不会在每个 epoch 重置。
