---
title: Add Weighter to Dataflex
createTime: 2025/06/30 19:19:16
permalink: /en/guide/weighter/tutorial/
icon: solar:add-circle-outline
---

# Add Weighter to Dataflex

## Weight Trainer Overview

Weight Trainer allows you to dynamically adjust sample weights during backpropagation based on sample importance at specific training stages.

**Working Mechanism**: The external Weight Trainer (`weight_trainer.py`) calls the `training_step` method of the Weighter component during the training loop to execute training. The `training_step` method is uniformly implemented by the base class `Weighter`, responsible for the complete process of forward propagation, loss calculation, and backpropagation. The specific weighting logic is completed by calling the `get_weighted_loss` method implemented by subclasses.

### Parameter Configuration

When using Weight Trainer, you need to add the following DataFlex-specific parameters to your `.yaml` configuration file:

```yaml
train_type: dynamic_weight   # Select trainer type. Available options:
                          # "dynamic_select" - Dynamic selection trainer
                          # "dynamic_mix" - Dynamic mixing trainer
                          # "dynamic_weight" - Dynamic weighting trainer
                          # "static" - Default static trainer
components_cfg_file: src/dataflex/configs/components.yaml
component_name: loss  # Select component name, corresponding to components defined in components_cfg_file
warmup_step: 1
train_step: 3 # Total training steps (including warm_up)
```

### Parameter Details

- `train_type`: Defines the training type. `dynamic_weight` enables Weight Trainer.
- `component_name`: Defines the specific strategy for data weighting. For example, `loss` uses a loss-based weighter.
- `components_cfg_file`: Defines the parameter file containing specific parameters for the corresponding strategy.
- `warmup_step`: Before the first dynamic weighting, the model needs to perform `warmup_step` steps of regular training. This helps the model establish initial understanding of data distribution.
- `train_step`: Total training steps (including warmup). Weight Trainer will dynamically weight samples at each training step after warmup completion.

## How to Add Custom Weighter in DataFlex

This document will use `custom_weighter` as an example to detail how to add and configure a custom sample weighter in the DataFlex framework for dynamic sample weight adjustment during training.

### Step 1: Create Weighter Implementation File

First, create a new Python file in the specified project path to implement the core logic of your custom weighter.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/weighter/custom_weighter.py`
2. **File Content**: In this file, define a new class `CustomWeighter` that inherits from `dataflex.train.weighter.base_weighter.Weighter`.

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
        Constructor for custom weighter
        
        Args:
            strategy: Weighting strategy, such as "uniform", "loss_based", etc.
            **kwargs: Other parameters passed to base class
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
        Core weighting logic.
        Calculate weighted total loss based on sample losses.
        
        Args:
            losses: Per-sample loss on current device (B,)
            ctx: Trainer context, can access global_step and other information
            model: Current model
            inputs: Input data
            
        Returns:
            Weighted total loss (scalar)
        """
        # Example logic: simple uniform weighting
        if not torch.is_tensor(losses) or losses.dim() == 0:
            return losses
            
        # Here you can implement your custom weighting strategy
        # For example: based on loss magnitude, gradient information, sample difficulty, etc.
        weights = torch.ones_like(losses) / losses.numel()
        weighted_loss = torch.sum(weights * losses)
        
        return weighted_loss
```

#### Key Points

- `@register_weighter('custom')`: This decorator registers your `CustomWeighter` class into the DataFlex framework with the unique name `custom`. This name will be used in configuration files.
- `CustomWeighter(Weighter)`: Your custom class must inherit from the framework's `Weighter` base class. The base class already implements the `training_step` method and `_per_sample_loss_from_logits` helper method.
- `__init__`: Constructor for necessary initialization. Call `super().__init__(**kwargs)` to ensure base class initialization is properly executed.
- `get_weighted_loss`: This is the core abstract method you need to implement to define the sample weighting algorithm. The base class's `training_step` method automatically calls this method to get weighted loss. The external Weight Trainer (`weight_trainer.py`) executes the complete training step including forward propagation, loss calculation, weighting, and backpropagation by calling the base class's `training_step` method.

### Step 2: Import New Module

To enable DataFlex framework to recognize and load your newly created weighter, edit the `__init__.py` file in the directory to expose your new module.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/weighter/__init__.py`
2. **Add Content**: Add the following line at the end of the file

```python
from .custom_weighter import CustomWeighter
```

### Step 3: Configure Weighter Parameters

Finally, define your new weighter and its parameters in the YAML configuration file for convenient use in experiments.

1. **File Path**: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. **Add Configuration**: Under the `weighters` configuration block, add a new entry for your `custom` weighter.

```yaml
weighters:
  # ...
  custom:
    name: custom
    params:
      strategy: uniform
  # ...
```

#### Key Points

- `params`: All parameters defined under this block will be passed as keyword arguments to the `__init__` constructor of the `CustomWeighter` class. For example, the `strategy` value here will be passed to the `strategy` parameter of the `__init__` method.