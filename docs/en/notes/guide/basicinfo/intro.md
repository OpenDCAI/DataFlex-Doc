---
title: Introduction
icon: mdi:tooltip-text-outline
createTime: 2025/06/13 14:36:47
permalink: /en/guide/intro/basicinfo/intro/
---
# Introduction

In recent years, the development of large-scale models has relied heavily on access to massive and high-quality training data. However, most mainstream training datasets and their processing pipelines remain unpublished, and the available open data resources are still limited in both scale and quality. This poses significant challenges for the broader community in constructing and optimizing training data for large models.

Although organizations such as Open-DataLab have made efforts to promote open-source datasets, data preparation remains a labor-intensive and fragmented process. At present, different teams often need to build their own data cleaning and construction pipelines from scratch, due to the lack of standardized and systematic tool support. Existing data processing frameworks (such as Hadoop and Spark) are primarily designed around traditional paradigms and have yet to effectively incorporate intelligent operators based on large language models (LLMs), resulting in limited support for efficiently building data at the scale and quality required for modern large model training.

To address these challenges, we propose **DataFlow** — a high-efficiency data preparation system composed of advanced operators and multi-stage data processing pipelines. DataFlow integrates rule-based methods, deep learning models, and LLMs to offer a modular, scalable, and reconfigurable design. It aims to improve the quality and efficiency of data cleaning, augmentation, and construction, thereby supporting the development of next-generation large models.

## DataFlow: A High-Quality Data Preparation System

**DataFlow** is a data evaluation and processing system designed to **clean, enhance, and evaluate** noisy data sources (e.g., PDFs, plain text, low-quality QA pairs) to produce high-quality training data. This curated data can then be used to improve large language model performance through targeted training approaches, including pre-training, supervised fine-tuning, and reinforcement learning, across both general domains (e.g., reasoning and retrieval) and specialized fields (e.g., healthcare, finance, and law).

Specifically, we have developed a diverse set of **operators**, built on rule-based methods, deep learning models, large language models (LLMs), and LLM APIs. These operators are systematically integrated into six independent data processing pipelines, collectively forming the complete **DataFlow system**.

In addition, we have developed an intelligent **DataFlow-Agent** capable of dynamically composing existing operators into new data processing workflows based on user requirements. This enables more flexible and efficient data construction and processing capabilities.
