---
title: Framework Design
icon: material-symbols:auto-transmission-sharp
createTime: 2025/06/13 14:59:56
permalink: /en/guide/basicinfo/framework/
---

# Framework Design

## Overview

DataFlex is an advanced dynamic training framework built on [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). It intelligently schedules data during training, supporting **dynamic sample selection**, **domain ratio adjustment**, and **dynamic weight allocation** to improve training efficiency and final model performance.

### Design Philosophy

The core design philosophy of DataFlex is: **Data-centric intelligent training scheduling**. Traditional training methods typically use fixed data order and ratios, while DataFlex allows models to dynamically adjust data usage strategies based on their current state during training, achieving more efficient learning. It is designed to seamlessly integrate with LlamaFactory, providing researchers and developers with more flexible and powerful training control capabilities.

## Core Architecture

### Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      LlamaFactory Framework                 │
├─────────────────────────────────────────────────────────────┤
│         Model Management · Data Processing · Optimizers     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Training Layer (DataFlex replaces LlamaFactory trainer)  │
│  ┌─────────────────┬─────────────────┬─────────────────────┐ │
│  │  Select Trainer │   Mix Trainer   │  Weight Trainer     │ │
│  │ (Dynamic Sample │  (Dynamic Ratio)│ (Dynamic Weights)   │ │
│  │   Selection)    │                 │                     │ │
│  ├─────────────────┼─────────────────┼─────────────────────┤ │
│  │ Selector Components│ Mixer Components│ Weighter Components│ │
│  │ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────────┐ │ │
│  │ │Loss Selector│ │ │Random Mixer │ │ │ Loss Weighter   │ │ │
│  │ │LESS Selector│ │ │Custom Mixer │ │ │ Custom Weighter │ │ │
│  │ │ Custom...   │ │ │   ...       │ │ │    ...          │ │ │
│  │ └─────────────┘ │ └─────────────┘ │ └─────────────────┘ │ │
│  └─────────────────┴─────────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

DataFlex adopts a modular design with the following layers:

1. **Base Layer (LlamaFactory)**: Provides model management, data processing, optimizers and other basic components
2. **Trainer Layer (DataFlex Trainers)**: **Replaces** LlamaFactory's original trainer, implementing three dynamic training modes
3. **Strategy Component Layer (Components)**: Provides specific data processing strategies (Selector/Mixer/Weighter)
4. **Registry System**: Manages component registration and loading

**Key Feature**: DataFlex doesn't add new layers on top of LlamaFactory, but **seamlessly replaces** its training layer, maintaining original functionality while enhancing training capabilities.

## Three Core Trainer Concepts

DataFlex provides three core trainers that can seamlessly integrate into LlamaFactory's training pipeline:

- **Select Trainer (Dynamic Selection Trainer)**: During training, dynamically selects a subset of samples from the dataset based on predefined strategies (Selector) for subsequent training, e.g., prioritizing "difficult" samples that the model finds challenging.
- **Mix Trainer (Dynamic Ratio Trainer)**: Supports dynamic adjustment of mixing ratios for data from different sources or domains during training.
- **Weight Trainer (Dynamic Weighting Trainer)**: Supports dynamic adjustment of sample weights during backpropagation, increasing learning intensity for model-preferred data.

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