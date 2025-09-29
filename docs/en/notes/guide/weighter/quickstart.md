---
title: Quick Start
createTime: 2025/06/30 19:19:16
permalink: /en/guide/weighter/quickstart/
icon: solar:bolt-outline
---

# Quick Start

The launch command is similar to [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). Below is an example using a loss weighter:

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/weighters/loss.yaml
```

Unlike vanilla LlamaFactory, your `.yaml` config file must include **DataFlex-specific parameters**:

```yaml
### dynamic_train
train_type: dynamic_weight 
components_cfg_file: src/dataflex/configs/components.yaml
component_name: loss
warmup_step: 1
train_step: 3 # total train steps (including warmup)
```