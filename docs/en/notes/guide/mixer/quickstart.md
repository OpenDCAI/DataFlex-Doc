---
title: Quick Start
createTime: 2025/06/30 19:19:16
permalink: /en/guide/mixer/quickstart/
icon: solar:bolt-outline
---

# Quick Start

The launch command is similar to [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). Below is an example using a random mixer:

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/mixers/random.yaml
```

Unlike vanilla LlamaFactory, your `.yaml` config file must include **DataFlex-specific parameters**:

```yaml
### dynamic_train
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: random
warmup_step: 100
update_step: 200
update_times: 3

eval_dataset: mmlu_eval
```