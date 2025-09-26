---
title: 简介
icon: mdi:tooltip-text-outline
createTime: 2025/06/13 14:51:34
permalink: /zh/guide/intro/basicinfo/intro/
---
# 简介

近年来，大模型的发展在很大程度上依赖于大规模且高质量的训练数据。首先，高质量数据集的准备至关重要，这一环节由我们在另一个项目 [DataFlow](https://github.com/OpenDCAI/DataFlow/tree/main) 中完成。在此基础上，训练阶段的数据与模型交互同样关键，例如：在训练过程中进行数据选择、配比，以及为不同数据分配权重。尽管学术界已经提出了若干基于 influence 等方法的数据选择策略，但始终缺乏一个统一、易用且可扩展的训练框架。

为了解决这一问题，我们基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 构建了 [DataFlex](https://github.com/OpenDCAI/DataFlex/tree/main)，一个以数据为中心、专注于优化训练过程中数据与模型交互的系统，兼具 **易用性** 与 **训练效果**。

## DataFlex：以数据为中心的模型训练系统

[DataFlex](https://github.com/OpenDCAI/DataFlex/tree/main) 在训练过程中提供了以下三大核心功能：

**1. 数据选择**
在训练过程中动态挑选对当前步骤最有利的数据，以提升训练效率与模型性能。

**2. 数据配比**
在多领域数据训练中，支持多种主流的数据比例混合方法，方便灵活地控制不同领域的数据分布。

**3. 数据权重分配**
针对每个 batch 的样本，计算其对模型的贡献度，并为不同数据点分配差异化权重，以实现更优的训练效果。

