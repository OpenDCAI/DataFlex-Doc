---
title: 快速开始
createTime: 2025/06/30 19:19:16
permalink: /zh/guide/mixer/quickstart/
icon: solar:bolt-outline
---

# 快速开始

启动命令类似于 [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory)。以下是使用随机混合器的示例：

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/mixers/random.yaml
```

与普通的 LlamaFactory 不同，您的 `.yaml` 配置文件必须包含 **DataFlex 特定的参数**：

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