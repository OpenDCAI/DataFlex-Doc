---
title: Introduction
icon: mdi:tooltip-text-outline
createTime: 2025/06/13 14:51:34
permalink: /en/guide/intro/basicinfo/intro/
---
# Introduction

In recent years, the development of large models has largely depended on large-scale, high-quality training data. First, the preparation of high-quality datasets is crucial, a process completed by our other project [DataFlow](https://github.com/OpenDCAI/DataFlow/tree/main). Building on this foundation, the interaction between data and models during training is equally important, such as data selection, mixing, and weighting throughout the training process. Although several influence-based methods have been proposed in academia — such as those based on the distributional distance between training and test data, as well as strategies like TracIn, Influence Function, and PMP — there still lacks a unified, user-friendly, and extensible training framework.

To address this problem, we built [DataFlex](https://github.com/OpenDCAI/DataFlex/tree/main) based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a data-centric system focused on optimizing data-model interactions during training, combining both **ease of use** and **training effectiveness**.

## DataFlex: A Data-Centric Model Training System

[DataFlex](https://github.com/OpenDCAI/DataFlex/tree/main) provides the following three core functionalities during training:

**1. Data Selection**
Dynamically select the most beneficial data for the current training step to improve training efficiency and model performance.

**2. Data Ratio Adjustment**
In multi-domain data training, support various mainstream data proportion mixing methods for flexible control of data distribution across different domains.

**3. Data Weight Allocation**
Calculate the contribution of each sample in a batch to the model and assign differentiated weights to different data points to achieve better training results.