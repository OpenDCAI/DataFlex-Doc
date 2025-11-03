---
title: 框架设计
icon: material-symbols:auto-transmission-sharp
createTime: 2025/06/13 14:59:56
permalink: /zh/guide/basicinfo/framework/
---

# 框架设计

## 概述

DataFlex 是一个基于 [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) 的以数据为中心的动态训练框架。它通过在训练过程中智能地调度数据，支持**动态样本选择**、**领域配比调整**和**动态权重分配**，旨在提升模型训练的效率与最终效果。

### 设计理念

DataFlex 的核心设计理念是：**以数据为中心的智能训练调度**。传统的训练方法通常采用固定的数据顺序和配比，而 DataFlex 允许模型在训练过程中根据其当前状态动态调整数据使用策略，从而实现更高效的学习。其设计思想是与 LlamaFactory 无缝集成，为研究者和开发者提供更灵活、更强大的训练控制能力。

在数据选择的过程中，往往需要对于数据样本进行Embedding, Inference, 计算梯度等操作。DataFlex旨在统一管理Embedding，大模型的推理和计算梯度等操作。

## 核心架构

### 整体架构图

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           LlamaFactory Framework                              │
├───────────────────────────────────────────────────────────────────────────────┤
│                  Model Management · Data Processing · Optimizers              │
├───────────────────────────────────────────────────────────────────────────────┤
│            Training Layer (DataFlex replaces LlamaFactory trainer)            │
│  ┌────────────────────────┬────────────────────────┬────────────────────────┐ │
│  │      Select Trainer    │       Mix Trainer      │     Weight Trainer     │ │
│  │   (Dynamic Selection)  │      (Dynamic Ratio)   │     (Dynamic Weights)  │ │
│  ├────────────────────────┼────────────────────────┼────────────────────────┤ │
│  │  Selector Components   │    Mixer Components    │   Weighter Components  │ │
│  │  ┌──────────────────┐  │  ┌──────────────────┐  │  ┌───────────────────┐ │ │
│  │  │  Loss Selector   │  │  │   Random Mixer   │  │  │   Loss Weighter   │ │ │
│  │  │  LESS Selector   │  │  │   Custom Mixer   │  │  │  Custom Weighter  │ │ │
│  │  │   Custom ...     │  │  │       ...        │  │  │        ...        │ │ │
│  │  └──────────────────┘  │  └──────────────────┘  │  └───────────────────┘ │ │
│  └────────────────────────┴────────────────────────┴────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 组件层次结构

DataFlex 采用模块化设计，主要包含以下层次：

1. **基础层 (LlamaFactory)**：提供模型管理、数据处理、优化器等基础组件
2. **训练器层 (DataFlex Trainers)**：**替换** LlamaFactory 原始训练器，实现样本选择，配比，以及Reweight三种动态训练模式
3. **策略组件层 (Components)**：提供具体的数据处理策略（Selector/Mixer/Weighter）
4. **注册系统 (Registry)**：管理组件的注册和加载

**关键特点**：DataFlex 不是在 LlamaFactory 上添加新层，而是**无缝替换**其训练层，保持原有基础功能的同时增强训练能力。

## 三大核心训练器概念

DataFlex 提供了三大核心训练器，它们都可以无缝接入 LlamaFactory 的训练流程：

- **Select Trainer (动态选择训练器)**: 在训练过程中，根据预设策略（Selector）动态地从数据集中挑选出一部分样本用于接下来的训练，例如优先训练模型认为"难"的样本。
- **Mix Trainer (动态配比训练器)**: 支持在训练中动态调整不同来源或领域数据的混合比例。
- **Weight Trainer (动态加权训练器)**: 支持在训练中动态调整样本反向传播时的权重，增大模型偏好的数据的学习力度。
