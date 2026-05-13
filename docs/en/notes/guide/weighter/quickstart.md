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
num_train_epochs: 1.0
train_step: 0 # set positive to fix total steps and override num_train_epochs
```

For multi-epoch runs, set `num_train_epochs: N` and keep `train_step: 0`. `warmup_step` is a global step threshold and does not reset each epoch.
