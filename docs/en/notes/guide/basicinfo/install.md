---
title: Installation
icon: material-symbols-light:download-rounded
createTime: 2025/06/09 10:29:31
permalink: /en/guide/install/
---
# Installation

Run the following commands to install:

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory
```

## Usage Example

The training command is very similar to LlamaFactory. Below is an example using LESS, refer to the paper for details [https://arxiv.org/abs/2402.04333](https://arxiv.org/abs/2402.04333):

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/less.yaml
```

**Note**: Unlike standard LlamaFactory, your `.yaml` configuration file must include DataFlex-specific parameters in addition to LlamaFactory's standard training parameters.

## Integration with LlamaFactory

DataFlex is fully compatible with LlamaFactory's configuration and usage:

1. **Configuration Compatibility**: Add DataFlex parameters on top of LlamaFactory configuration
2. **Consistent Commands**: Use `dataflex-cli` instead of `llamafactory-cli`
3. **Feature Preservation**: Supports all original LlamaFactory functionality
4. **Seamless Switching**: Can fallback to original training mode with `train_type: static`

This design ensures users can progressively adopt DataFlex functionality without major modifications to existing workflows.