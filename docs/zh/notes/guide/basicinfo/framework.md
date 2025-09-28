---
title: 框架设计
icon: material-symbols:auto-transmission-sharp
createTime: 2025/06/13 14:59:56
permalink: /zh/guide/basicinfo/framework/
---

# 框架设计

## 概述

DataFlex 是一个基于 [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) 的高级动态训练框架。它通过在训练过程中智能地调度数据，支持**动态样本选择**、**领域配比调整**和**动态权重分配**，旨在提升模型训练的效率与最终效果。

### 设计理念

DataFlex 的核心设计理念是：**以数据为中心的智能训练调度**。传统的训练方法通常采用固定的数据顺序和配比，而 DataFlex 允许模型在训练过程中根据其当前状态动态调整数据使用策略，从而实现更高效的学习。其设计思想是与 LlamaFactory 无缝集成，为研究者和开发者提供更灵活、更强大的训练控制能力。

## 核心架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      LlamaFactory 框架                      │
├─────────────────────────────────────────────────────────────┤
│              模型管理 · 数据处理 · 优化器等基础组件              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    训练层 (DataFlex 替换 LlamaFactory 原始训练器)             │
│  ┌─────────────────┬─────────────────┬─────────────────────┐ │
│  │  Select Trainer │   Mix Trainer   │  Weight Trainer     │ │
│  │  (动态样本选择)  │   (动态配比)     │   (动态权重)        │ │
│  ├─────────────────┼─────────────────┼─────────────────────┤ │
│  │  Selector 组件  │   Mixer 组件    │  Weighter 组件      │ │
│  │ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────────┐ │ │
│  │ │Loss Selector│ │ │Random Mixer │ │ │ Loss Weighter   │ │ │
│  │ │LESS Selector│ │ │Custom Mixer │ │ │ Custom Weighter │ │ │
│  │ │ Custom...   │ │ │   ...       │ │ │    ...          │ │ │
│  │ └─────────────┘ │ └─────────────┘ │ └─────────────────┘ │ │
│  └─────────────────┴─────────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 组件层次结构

DataFlex 采用模块化设计，主要包含以下层次：

1. **基础层 (LlamaFactory)**：提供模型管理、数据处理、优化器等基础组件
2. **训练器层 (DataFlex Trainers)**：**替换** LlamaFactory 原始训练器，实现三种动态训练模式
3. **策略组件层 (Components)**：提供具体的数据处理策略（Selector/Mixer/Weighter）
4. **注册系统 (Registry)**：管理组件的注册和加载

**关键特点**：DataFlex 不是在 LlamaFactory 上添加新层，而是**无缝替换**其训练层，保持原有基础功能的同时增强训练能力。

## 三大核心训练器概念

DataFlex 提供了三大核心训练器，它们都可以无缝接入 LlamaFactory 的训练流程：

- **Select Trainer (动态选择训练器)**: 在训练过程中，根据预设策略（Selector）动态地从数据集中挑选出一部分样本用于接下来的训练，例如优先训练模型认为"难"的样本。
- **Mix Trainer (动态配比训练器)**: 支持在训练中动态调整不同来源或领域数据的混合比例。
- **Weight Trainer (动态加权训练器)**: 支持在训练中动态调整样本反向传播时的权重，增大模型偏好的数据的学习力度。

## 使用示例

启动训练的命令与 LlamaFactory 非常相似。以下是一个使用 LESS 的示例，具体原理参考论文 [https://arxiv.org/abs/2402.04333](https://arxiv.org/abs/2402.04333)：

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/less.yaml
```

**注意**：与标准 LlamaFactory 不同，您的 `.yaml` 配置文件中除了需要包含 LlamaFactory 的标准训练参数外，还必须指定 DataFlex 的特定参数。


## 与 LlamaFactory 的集成

DataFlex 完全兼容 LlamaFactory 的配置和使用方式：

1. **配置兼容**：在 LlamaFactory 配置基础上添加 DataFlex 参数
2. **命令一致**：使用 `dataflex-cli` 替代 `llamafactory-cli`
3. **功能保持**：支持所有 LlamaFactory 的原有功能
4. **无缝切换**：可以通过 `train_type: static` 回退到原始训练模式

这种设计确保了用户可以渐进式地采用 DataFlex 的功能，无需对现有工作流进行大幅修改。
