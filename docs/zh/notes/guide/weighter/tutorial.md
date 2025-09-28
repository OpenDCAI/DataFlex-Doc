---
title: 向Dataflex添加Weighter
createTime: 2025/06/30 19:19:16
permalink: /zh/guide/weighter/tutorial/
icon: solar:add-circle-outline
---

# 向Dataflex添加Weighter

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

## 如何在 DataFlex 中添加自定义 Weighter

本文档将以 `custom_weighter` 为例，详细介绍如何在 DataFlex 框架中添加并配置一个自定义的样本加权器，实现训练过程中的动态样本权重调整。

### 步骤一：创建加权器实现文件

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

#### 关键点说明

- `@register_weighter('custom')`: 这是一个装饰器，用于将您的 `CustomWeighter` 类注册到 DataFlex 框架中，并赋予其一个唯一的名称 `custom`。这个名称将在后续的配置文件中使用。
- `CustomWeighter(Weighter)`: 您的自定义类必须继承自框架提供的 `Weighter` 基类。基类已经实现了 `training_step` 方法和 `_per_sample_loss_from_logits` 辅助方法。
- `__init__`: 构造函数用于执行必要的初始化操作。调用 `super().__init__(**kwargs)` 来确保基类的初始化逻辑被正确执行。
- `get_weighted_loss`: 这是您需要实现的核心抽象方法，用于定义样本加权算法。基类的 `training_step` 方法会自动调用此方法来获取加权后的损失。外部 Weight Trainer (`weight_trainer.py`) 通过调用基类的 `training_step` 方法来执行完整的训练步骤，包括前向传播、损失计算、加权处理和反向传播。

### 步骤二：导入新模块

为了让 DataFlex 框架能够识别并加载您新创建的加权器，需要编辑该目录下的 `__init__.py` 文件，以暴露您的新模块。

1. **文件路径**: `DataFlex-Preview/src/dataflex/train/weighter/__init__.py`
2. **添加内容**: 在文件末尾添加以下行

```python
from .custom_weighter import CustomWeighter
```

### 步骤三：配置加权器参数

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

#### 关键点说明

- `params`: 该块下定义的所有参数都将作为关键字参数传递给 `CustomWeighter` 类的 `__init__` 构造函数。例如，这里的 `strategy` 值会传递给 `__init__` 方法的 `strategy` 参数。