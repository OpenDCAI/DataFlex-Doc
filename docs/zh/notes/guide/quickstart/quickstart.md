---
title: 快速开始
createTime: 2025/06/30 19:19:16
permalink: /zh/guide/quickstart/
icon: solar:flag-2-broken
---

# 快速开始

启动命令类似于 [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory)。以下是使用 [LESS](https://arxiv.org/abs/2402.04333) 的示例：

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/less.yaml
```

与普通的 LlamaFactory 不同，您的 `.yaml` 配置文件必须包含 **DataFlex 特定的参数**。
