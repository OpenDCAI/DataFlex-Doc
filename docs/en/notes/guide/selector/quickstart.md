---
title: Quick Start
createTime: 2025/06/30 19:19:16
permalink: /en/guide/selector/quickstart/
icon: solar:bolt-outline
---

# Quick Start

The launch command is similar to [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). Below is an example using [LESS](https://arxiv.org/abs/2402.04333):

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/less.yaml
```

Unlike vanilla LlamaFactory, your `.yaml` config file must include **DataFlex-specific parameters**:

```yaml
### dynamic_train
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: less
warmup_step: 4
update_step: 3
update_times: 2
num_train_epochs: 1.0
train_step: 0

eval_dataset: alpaca_zh_demo
```

`update_times` is the number of dynamic selections per Flex epoch. Use `num_train_epochs: 1.0` for one Flex epoch; use `num_train_epochs: N` with `train_step: 0` for multi-epoch runs. If `train_step > 0`, it fixes the total number of steps and overrides `num_train_epochs`.
