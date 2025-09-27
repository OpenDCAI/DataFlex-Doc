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
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/less.yaml
```

**注意**：与标准 LlamaFactory 不同，您的 `.yaml` 配置文件中除了需要包含 LlamaFactory 的标准训练参数外，还必须指定 DataFlex 的特定参数。

## Select Trainer 详解

Select Trainer 允许您在训练的特定阶段，根据模型的当前状态动态调整后续的训练数据顺序。

### 参数配置

当使用 Select Trainer 时，需要在 `.yaml` 配置文件中添加以下 DataFlex 特定参数：

```yaml
# Select Trainer 参数
train_type: dynamic_select   # 必须，指定使用 Select Trainer
                             # 从 [dynamic_select, dynamic_mix, dynamic_weight, static] 中选择
                             # 分别对应动态选择、动态混合、动态加权和静态（即 llamafactory 原始训练流程）
component_name: loss         # 必须，指定具体使用的选择器策略，如 loss 或 less
components_cfg_file: src/dataflex/configs/components.yaml    # 必须，指定具体使用的选择器的参数
warmup_step: 200             # 必选，在首次数据选择前，模型预热训练的步数
update_step: 1000            # 必须，每隔多少步（step）执行一次数据选择
update_times: 2              # 必须，数据选择执行的总次数
```

### 参数详解

- `train_type`: 定义训练类型。`dynamic_select` 表示启用 Select Trainer。
- `component_name`: 定义数据选择的具体策略。例如，`loss` 表示使用基于损失值的选择器。
- `components_cfg_file`: 定义数据选择策略的参数文件，包含对应策略的特定参数。
- `warmup_step`: 在执行第一次动态选择之前，模型需要先进行 `warmup_step` 步的常规训练。这有助于模型建立对数据分布的初步认知。
- `update_step`: 数据选择的频率。每当训练进行 `update_step` 步后，Selector 将被触发，重新选择 `update_step * global_batch_size` 个样本用于下一阶段的训练。
- `update_times`: 整个训练过程中，动态数据选择执行的总次数。因此总的训练步数为 `(update_times * update_step + warmup_step) * global_batch_size`

### 如何在 DataFlex 中添加自定义选择器

本文档将以 `custom_selector` 为例，详细介绍如何在 DataFlex 框架中添加并配置一个自定义的数据选择器，实现训练过程中的动态数据点选择。

#### 步骤一：创建选择器实现文件

首先，在项目指定路径下创建一个新的 Python 文件，用于实现自定义选择器的核心逻辑。

1. **文件路径**: `DataFlex-Preview/src/dataflex/train/selector/custom_selector.py`
2. **文件内容**: 在该文件中，定义一个继承自 `dataflex.train.selector.base_selector.Selector` 的新类 `CustomSelector`。

```python
from dataflex.core.registry import register_selector
from .base_selector import logger, Selector

@register_selector('custom')
class CustomSelector(Selector):
    """
    一个自定义数据选择器的示例实现。
    """
    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
    ):
        """
        构造函数，用于初始化选择器。
        """
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        logger.info(f"CustomSelector initialized.")

    def select(self, model, step_id: int, num_samples: int, **kwargs):
        """
        核心选择逻辑。
        此方法定义了如何从数据集中选择样本。

        Args:
            model: 当前的模型。
            step_id (int): 当前的训练步数。
            num_samples (int): 需要选择的样本数量。

        Returns:
            list: 包含被选中样本索引的列表。
        """
        # 示例逻辑：简单返回从 0 到 num_samples-1 的索引列表。
        # 您可以在此实现更复杂的选择算法。
        return list(range(num_samples))
```

**关键点说明**:

- `@register_selector('custom')`: 这是一个装饰器，用于将您的 `CustomSelector` 类注册到 DataFlex 框架中，并赋予其一个唯一的名称 `custom`。这个名称将在后续的配置文件中使用。
- `CustomSelector(Selector)`: 您的自定义类必须继承自框架提供的 `Selector` 基类。
- `__init__`: 构造函数用于执行必要的初始化操作。调用 `super().__init__(...)` 来确保基类的初始化逻辑被正确执行。
- `select`: 这是实现数据选择算法的核心方法。您需要根据自己的需求重写此方法。
- `warmup` (可选): 您还可以根据需要重写 `warmup` 方法，用于选择用于 warmup 的数据。默认随机采样数据用于 warmup 阶段训练。

#### 步骤二：导入新模块

为了让 DataFlex 框架能够识别并加载您新创建的选择器，需要编辑该目录下的 `__init__.py` 文件，以暴露您的新模块。

1. **文件路径**: `DataFlex-Preview/src/dataflex/train/selector/__init__.py`
2. **添加内容**: 在文件末尾添加以下行，以导入 `CustomSelector` 类。

```python
from .custom_selector import *
```

#### 步骤三：配置选择器参数

最后，在 YAML 配置文件中定义您的新选择器及其参数，以便在实验中方便地调用。

1. **文件路径**: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. **添加配置**: 在 `selectors` 配置块下，为您的 `custom` 选择器添加新的条目。

```yaml
selectors:
  # ...
  # 添加您的自定义选择器配置
  custom:
    name: custom
    params:
      cache_dir: ../dataflex_saves/custom_output
  # ...
```

**关键点说明**:

- `params`: 该块下定义的所有参数都将作为关键字参数传递给 `CustomSelector` 类的 `__init__` 构造函数。例如，这里的 `cache_dir` 值会传递给 `__init__` 方法的 `cache_dir` 参数。

## Mix Trainer 详解

Mix Trainer 允许您在训练的特定阶段，根据模型的当前状态动态调整后续的领域数据配比。

### 参数配置

当使用 Mix Trainer 时，需要在 `.yaml` 配置文件中添加以下 DataFlex 特定参数：

```yaml
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: random
mixture_sample_rule: mixture     # 初始采样规则，mixture为根据init_mixture_proportions比例混合（可动态调整），
                                 # stratified为固定按源数据集大小比例分层，uniform为固定均匀分布
init_mixture_proportions: [0.7, 0.3]  # 对应初始的比例，如果mixture_sample_rule为mixture必须设置
warmup_step: 4
update_step: 3
update_times: 2
```

### 参数详解

- `train_type`: 定义训练类型。`dynamic_mix` 表示启用 Mix Trainer。
- `component_name`: 定义数据选择的具体策略。例如，`random` 表示使用随机的领域配比器。
- `components_cfg_file`: 定义策略的参数文件，包含对应策略的特定参数。
- `mixture_sample_rule`: 初始采样规则，必选，`mixture` 为根据 `init_mixture_proportions` 比例混合（可动态调整），`stratified` 为固定按源数据集大小比例分层，`uniform` 为固定均匀分布。
- `init_mixture_proportions`: 初始采样对应的比例，`mixture_sample_rule='mixture'` 时需要指定。
- `warmup_step`: 在执行第一次动态配比更新前，模型需要先进行 `warmup_step` 步的常规训练。这有助于模型建立对数据分布的初步认知。
- `update_step`: 领域配比更新的频率。每当训练进行 `update_step` 步后，Mixer 将被触发，更新领域配比用于下一阶段的训练。
- `update_times`: 整个训练过程中，动态数据配比计算的总次数。因此总的训练步数为 `(update_times * update_step + warmup_step) * global_batch_size`

### 静态混合配置

Mix Trainer 支持静态混合模式，通过设置 `static_mix: true` 来固定初始比例：

```yaml
train_type: dynamic_mix
static_mix: true                      # 是否固定初始静态混合比例（仅在dynamic_mix训练器中生效）
mixture_sample_rule: mixture          # 初始采样规则
init_mixture_proportions: [0.7, 0.3]  # 对应初始的比例，可通过额外算法自行调整
train_step: 3                         # 总训练步数（仅在dynamic_mix训练器中生效），不考虑warmup和update steps
```

启用静态混合后，训练过程中将使用固定的 `init_mixture_proportions` 比例，不再动态调整。

### 如何在 DataFlex 中添加自定义 Mixer

本文档将以 `random_mixer` 为例，详细介绍如何在 DataFlex 框架中添加并配置一个自定义的数据配比器，实现训练过程中的动态领域配比。

#### 步骤一：创建配比器实现文件

首先，在项目指定路径下创建一个新的 Python 文件，用于实现自定义配比器的核心逻辑。

1. **文件路径**: `DataFlex-Preview/src/dataflex/train/mixer/random_mixer.py`
2. **文件内容**: 在该文件中，定义一个继承自 `dataflex.train.mixer.base_mixer.Mixer` 的新类 `RandomMixer`。

```python
from dataflex.core.registry import register_mixer
from dataflex.utils.logging import logger
from .base_mixer import Mixer

import numpy as np

@register_mixer("random")
class RandomMixer(Mixer):
    def __init__(self, mixture_manager, seed):
        super().__init__(mixture_manager)
        self.seed = seed
    
    def mix(self, model, step_id: int, **kwargs) -> np.ndarray:
        """
        随机生成一组比例向量。

        Returns:
            np.ndarray: 长度为源数量的归一化比例数组。
        """
        k = len(self.mixture_manager.names)
        np.random.seed(self.seed)
        raw = np.random.random(k)
        probs = raw / raw.sum()  # 归一化
        logger.info(f"[RandomMixer] Step {step_id} Generated proportions: {probs}")

        return probs
```

**关键点说明**:

- `@register_mixer('random')`: 这是一个装饰器，用于将您的 `RandomMixer` 类注册到 DataFlex 框架中，并赋予其一个唯一的名称 `random`。这个名称将在后续的配置文件中使用。
- `RandomMixer(Mixer)`: 您的自定义类必须继承自框架提供的 `Mixer` 基类。
- `__init__`: 构造函数用于执行必要的初始化操作。调用 `super().__init__(...)` 来确保基类的初始化逻辑被正确执行。
- `mix`: 这是实现数据配比算法的核心方法。您需要根据自己的需求重写此方法，需要返回长度为源数量的归一化比例数组。

#### 步骤二：导入新模块

为了让 DataFlex 框架能够识别并加载您新创建的配比器，需要编辑该目录下的 `__init__.py` 文件，以暴露您的新模块。

1. **文件路径**: `DataFlex-Preview/src/dataflex/train/mixer/__init__.py`
2. **添加内容**: 在文件末尾添加以下行

```python
from .random_mixer import RandomMixer
```

#### 步骤三：配置配比器参数

最后，在 YAML 配置文件中定义您的新配比器及其参数，以便在实验中方便地调用。

1. **文件路径**: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. **添加配置**: 在 `mixers` 配置块下，为您的 `random` 配比器添加新的条目。

```yaml
mixers:
  # ...
  random:
    name: random
    params:
      seed: 42
  # ...
```

**关键点说明**:

- `params`: 该块下定义的所有参数都将作为关键字参数传递给 `RandomMixer` 类的 `__init__` 构造函数。例如，这里的 `seed` 值会传递给 `__init__` 方法的 `seed` 参数。

## Weight Trainer 详解

Weight Trainer 允许您在训练的特定阶段，根据样本的重要性动态调整样本在反向传播时的权重。

**工作机制**：外部 Weight Trainer (`weight_trainer.py`) 在训练循环中调用 Weighter 组件的 `training_step` 方法执行训练。`training_step` 方法由基类 `Weighter` 统一实现，负责前向传播、损失计算和反向传播的完整流程。具体的加权逻辑通过调用子类实现的 `get_weighted_loss` 方法来完成。

### 参数配置

当使用 Weight Trainer 时，需要在 `.yaml` 配置文件中添加以下 DataFlex 特定参数：

```yaml
train_type: dynamic_weight   # 选择训练器类型。可选值包括：
                          # "dynamic_select" - 动态选择训练器
                          # "dynamic_mix" - 动态混合训练器
                          # "dynamic_weight" - 动态加权训练器
                          # "static" - 默认静态训练器
components_cfg_file: src/dataflex/configs/components.yaml
component_name: loss  # 选择组件名称，对应 components_cfg_file 中定义的组件
warmup_step: 1
train_step: 3 # 总训练步数（包括warm_up）
```

### 参数详解

- `train_type`: 定义训练类型。`dynamic_weight` 表示启用 Weight Trainer。
- `component_name`: 定义数据加权的具体策略。例如，`loss` 表示使用基于损失值的加权器。
- `components_cfg_file`: 定义策略的参数文件，包含对应策略的特定参数。
- `warmup_step`: 在执行第一次动态加权前，模型需要先进行 `warmup_step` 步的常规训练。这有助于模型建立对数据分布的初步认知。
- `train_step`: 总训练步数（包括 warmup），Weight Trainer 将在 warmup 完成后的每个训练步骤中对样本进行动态加权。

### 如何在 DataFlex 中添加自定义 Weighter

本文档将以 `custom_weighter` 为例，详细介绍如何在 DataFlex 框架中添加并配置一个自定义的样本加权器，实现训练过程中的动态样本权重调整。

#### 步骤一：创建加权器实现文件

首先，在项目指定路径下创建一个新的 Python 文件，用于实现自定义加权器的核心逻辑。

1. **文件路径**: `DataFlex-Preview/src/dataflex/train/weighter/custom_weighter.py`
2. **文件内容**: 在该文件中，定义一个继承自 `dataflex.train.weighter.base_weighter.Weighter` 的新类 `CustomWeighter`。

```python
from dataflex.core.registry import register_weighter
from dataflex.utils.logging import logger
from typing import Any, Union
from torch import nn
import torch
from .base_weighter import Weighter

@register_weighter("custom")
class CustomWeighter(Weighter):
    def __init__(self, strategy: str = "uniform", **kwargs):
        """
        自定义加权器的构造函数
        
        Args:
            strategy: 加权策略，如 "uniform"、"loss_based" 等
            **kwargs: 传递给基类的其他参数
        """
        super().__init__(**kwargs)
        self.strategy = strategy
        logger.info(f"CustomWeighter initialized with strategy: {strategy}")
    
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        *,
        ctx: Any = None,
        model: nn.Module | None = None,
        inputs: dict[str, Union[torch.Tensor, Any]] | None = None,
    ) -> torch.Tensor:
        """
        核心加权逻辑。
        根据样本损失计算加权后的总损失。
        
        Args:
            losses: 本卡的 per-sample loss (B,)
            ctx: Trainer 上下文，可获取 global_step 等信息
            model: 当前模型
            inputs: 输入数据
            
        Returns:
            加权后的总损失（标量）
        """
        # 示例逻辑：简单的均匀加权
        if not torch.is_tensor(losses) or losses.dim() == 0:
            return losses
            
        # 这里可以实现您的自定义加权策略
        # 例如：基于损失大小、梯度信息、样本难度等
        weights = torch.ones_like(losses) / losses.numel()
        weighted_loss = torch.sum(weights * losses)
        
        return weighted_loss
```

**关键点说明**:

- `@register_weighter('custom')`: 这是一个装饰器，用于将您的 `CustomWeighter` 类注册到 DataFlex 框架中，并赋予其一个唯一的名称 `custom`。这个名称将在后续的配置文件中使用。
- `CustomWeighter(Weighter)`: 您的自定义类必须继承自框架提供的 `Weighter` 基类。基类已经实现了 `training_step` 方法和 `_per_sample_loss_from_logits` 辅助方法。
- `__init__`: 构造函数用于执行必要的初始化操作。调用 `super().__init__(**kwargs)` 来确保基类的初始化逻辑被正确执行。
- `get_weighted_loss`: 这是您需要实现的核心抽象方法，用于定义样本加权算法。基类的 `training_step` 方法会自动调用此方法来获取加权后的损失。外部 Weight Trainer (`weight_trainer.py`) 通过调用基类的 `training_step` 方法来执行完整的训练步骤，包括前向传播、损失计算、加权处理和反向传播。

#### 步骤二：导入新模块

为了让 DataFlex 框架能够识别并加载您新创建的加权器，需要编辑该目录下的 `__init__.py` 文件，以暴露您的新模块。

1. **文件路径**: `DataFlex-Preview/src/dataflex/train/weighter/__init__.py`
2. **添加内容**: 在文件末尾添加以下行

```python
from .custom_weighter import CustomWeighter
```

#### 步骤三：配置加权器参数

最后，在 YAML 配置文件中定义您的新加权器及其参数，以便在实验中方便地调用。

1. **文件路径**: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. **添加配置**: 在 `weighters` 配置块下，为您的 `custom` 加权器添加新的条目。

```yaml
weighters:
  # ...
  custom:
    name: custom
    params:
      strategy: uniform
  # ...
```

**关键点说明**:

- `params`: 该块下定义的所有参数都将作为关键字参数传递给 `CustomWeighter` 类的 `__init__` 构造函数。例如，这里的 `strategy` 值会传递给 `__init__` 方法的 `strategy` 参数。

## 与 LlamaFactory 的集成

DataFlex 完全兼容 LlamaFactory 的配置和使用方式：

1. **配置兼容**：在 LlamaFactory 配置基础上添加 DataFlex 参数
2. **命令一致**：使用 `dataflex-cli` 替代 `llamafactory-cli`
3. **功能保持**：支持所有 LlamaFactory 的原有功能
4. **无缝切换**：可以通过 `train_type: static` 回退到原始训练模式

这种设计确保了用户可以渐进式地采用 DataFlex 的功能，无需对现有工作流进行大幅修改。
