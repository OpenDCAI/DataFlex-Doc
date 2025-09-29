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
train_step: 3 # 总训练步数（包括warm_up） 
```