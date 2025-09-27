---
title: Quick Start
createTime: 2025/06/30 19:19:16
permalink: /en/guide/quickstart/
icon: solar:flag-2-broken
---

# Quick Start

The launch command is similar to [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory).
Below is an example using [LESS](https://arxiv.org/abs/2402.04333) :

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/less.yaml
```

Unlike vanilla LlamaFactory, your `.yaml` config file must also include **DataFlex-specific parameters**.