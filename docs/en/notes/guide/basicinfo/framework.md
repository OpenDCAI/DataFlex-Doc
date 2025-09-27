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
│                     LlamaFactory Framework                  │
├─────────────────────────────────────────────────────────────┤
│            Model Management · Data Processing · Optimizers   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Training Layer (DataFlex replaces LlamaFactory trainers) │
│  ┌─────────────────┬─────────────────┬─────────────────────┐ │
│  │  Select Trainer │   Mix Trainer   │  Weight Trainer     │ │
│  │ (Dynamic Select) │ (Dynamic Ratio) │ (Dynamic Weight)    │ │
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

1. **Foundation Layer (LlamaFactory)**: Provides model management, data processing, optimizers, and other basic components
2. **Trainer Layer (DataFlex Trainers)**: **Replaces** LlamaFactory's original trainers, implementing three dynamic training modes
3. **Strategy Component Layer (Components)**: Provides specific data processing strategies (Selector/Mixer/Weighter)
4. **Registry System**: Manages component registration and loading

**Key Feature**: DataFlex doesn't add new layers on top of LlamaFactory, but **seamlessly replaces** its training layer, maintaining original functionalities while enhancing training capabilities.

## Three Core Trainer Concepts

DataFlex provides three core trainers that can seamlessly integrate into LlamaFactory's training pipeline:

- **Select Trainer (Dynamic Selection Trainer)**: During training, dynamically selects a subset of samples from the dataset based on predefined strategies (Selector) for subsequent training, such as prioritizing "difficult" samples as perceived by the model.
- **Mix Trainer (Dynamic Ratio Trainer)**: Supports dynamic adjustment of mixing ratios between different sources or domain data during training.
- **Weight Trainer (Dynamic Weighting Trainer)**: Supports dynamic adjustment of sample weights during backpropagation, increasing learning intensity for model-preferred data.

## Usage Example

The training command is very similar to LlamaFactory. Here's an example using LESS, see paper [https://arxiv.org/abs/2402.04333](https://arxiv.org/abs/2402.04333) for details:

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/less.yaml
```

**Note**: Unlike standard LlamaFactory, your `.yaml` configuration file must include DataFlex-specific parameters in addition to LlamaFactory's standard training parameters.

## Select Trainer Details

Select Trainer allows you to dynamically adjust the order of subsequent training data based on the model's current state at specific training stages.

### Parameter Configuration

When using Select Trainer, add the following DataFlex-specific parameters to your `.yaml` configuration file:

```yaml
# Select Trainer parameters
train_type: dynamic_select   # Required, specify Select Trainer
                             # Choose from [dynamic_select, dynamic_mix, dynamic_weight, static]
                             # Corresponding to dynamic selection, dynamic mixing, dynamic weighting, and static (original LlamaFactory training)
component_name: loss         # Required, specify the selector strategy, such as loss or less
components_cfg_file: src/dataflex/configs/components.yaml    # Required, specify selector parameters
warmup_step: 200             # Required, warmup training steps before first data selection
update_step: 1000            # Required, interval steps for data selection execution
update_times: 2              # Required, total number of data selection executions
```

### Parameter Details

- `train_type`: Defines training type. `dynamic_select` enables Select Trainer.
- `component_name`: Defines specific data selection strategy. For example, `loss` uses loss-based selector.
- `components_cfg_file`: Defines parameter file for data selection strategy, containing specific parameters for the corresponding strategy.
- `warmup_step`: Before the first dynamic selection, the model needs `warmup_step` steps of regular training. This helps the model establish initial understanding of data distribution.
- `update_step`: Data selection frequency. After every `update_step` steps, the Selector is triggered to reselect `update_step * global_batch_size` samples for the next training phase.
- `update_times`: Total number of dynamic data selection executions during training. Therefore, total training steps = `(update_times * update_step + warmup_step) * global_batch_size`

### How to Add Custom Selector in DataFlex

This document uses `custom_selector` as an example to detail how to add and configure a custom data selector in the DataFlex framework for dynamic data point selection during training.

#### Step 1: Create Selector Implementation File

First, create a new Python file in the specified project path to implement the core logic of the custom selector.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/selector/custom_selector.py`
2. **File Content**: Define a new class `CustomSelector` that inherits from `dataflex.train.selector.base_selector.Selector`.

```python
from dataflex.core.registry import register_selector
from .base_selector import logger, Selector

@register_selector('custom')
class CustomSelector(Selector):
    """
    A custom data selector implementation example.
    """
    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
    ):
        """
        Constructor for initializing the selector.
        """
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        logger.info(f"CustomSelector initialized.")

    def select(self, model, step_id: int, num_samples: int, **kwargs):
        """
        Core selection logic.
        This method defines how to select samples from the dataset.

        Args:
            model: Current model.
            step_id (int): Current training step.
            num_samples (int): Number of samples to select.

        Returns:
            list: List containing indices of selected samples.
        """
        # Example logic: simply return indices from 0 to num_samples-1
        # You can implement more complex selection algorithms here
        return list(range(num_samples))
```

**Key Points**:

- `@register_selector('custom')`: This decorator registers your `CustomSelector` class in the DataFlex framework with the unique name `custom`. This name will be used in subsequent configuration files.
- `CustomSelector(Selector)`: Your custom class must inherit from the framework's `Selector` base class.
- `__init__`: Constructor for necessary initialization. Call `super().__init__(...)` to ensure base class initialization logic is properly executed.
- `select`: Core method for implementing data selection algorithm. You need to override this method according to your requirements.
- `warmup` (optional): You can also override the `warmup` method for selecting warmup data. By default, random sampling is used for warmup phase training.

#### Step 2: Import New Module

To enable the DataFlex framework to recognize and load your newly created selector, edit the `__init__.py` file in the same directory to expose your new module.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/selector/__init__.py`
2. **Add Content**: Add the following line at the end of the file to import the `CustomSelector` class.

```python
from .custom_selector import *
```

#### Step 3: Configure Selector Parameters

Finally, define your new selector and its parameters in the YAML configuration file for convenient use in experiments.

1. **File Path**: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. **Add Configuration**: Add a new entry for your `custom` selector under the `selectors` configuration block.

```yaml
selectors:
  # ...
  # Add your custom selector configuration
  custom:
    name: custom
    params:
      cache_dir: ../dataflex_saves/custom_output
  # ...
```

**Key Points**:

- `params`: All parameters defined under this block will be passed as keyword arguments to the `__init__` constructor of the `CustomSelector` class. For example, the `cache_dir` value here will be passed to the `cache_dir` parameter of the `__init__` method.

## Mix Trainer Details

Mix Trainer allows you to dynamically adjust the ratio of domain data for subsequent training based on the model's current state at specific training stages.

### Parameter Configuration

When using Mix Trainer, add the following DataFlex-specific parameters to your `.yaml` configuration file:

```yaml
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: random
mixture_sample_rule: mixture     # Initial sampling rule: mixture uses init_mixture_proportions ratios (dynamically adjustable),
                                 # stratified uses fixed stratified sampling by source dataset size, uniform uses fixed uniform distribution
init_mixture_proportions: [0.7, 0.3]  # Corresponding initial ratios, must be set if mixture_sample_rule is mixture
warmup_step: 4
update_step: 3
update_times: 2
```

### Parameter Details

- `train_type`: Defines training type. `dynamic_mix` enables Mix Trainer.
- `component_name`: Defines specific data selection strategy. For example, `random` uses random domain ratio mixer.
- `components_cfg_file`: Defines parameter file for the strategy, containing specific parameters for the corresponding strategy.
- `mixture_sample_rule`: Initial sampling rule, required. `mixture` uses `init_mixture_proportions` ratios (dynamically adjustable), `stratified` uses fixed stratified sampling by source dataset size, `uniform` uses fixed uniform distribution.
- `init_mixture_proportions`: Corresponding initial sampling ratios, must be specified when `mixture_sample_rule='mixture'`.
- `warmup_step`: Before the first dynamic ratio update, the model needs `warmup_step` steps of regular training. This helps the model establish initial understanding of data distribution.
- `update_step`: Domain ratio update frequency. After every `update_step` steps, the Mixer is triggered to update domain ratios for the next training phase.
- `update_times`: Total number of dynamic data ratio calculations during training. Therefore, total training steps = `(update_times * update_step + warmup_step) * global_batch_size`

### Static Mix Configuration

Mix Trainer supports static mix mode by setting `static_mix: true` to fix initial ratios:

```yaml
train_type: dynamic_mix
static_mix: true                      # Whether to fix initial static mix ratios (only effective in dynamic_mix trainer)
mixture_sample_rule: mixture          # Initial sampling rule
init_mixture_proportions: [0.7, 0.3]  # Corresponding initial ratios, can be adjusted by additional algorithms
train_step: 3                         # Total training steps (only effective in dynamic_mix trainer), excluding warmup and update steps
```

When static mix is enabled, fixed `init_mixture_proportions` ratios are used throughout training without dynamic adjustment.

### How to Add Custom Mixer in DataFlex

This document uses `random_mixer` as an example to detail how to add and configure a custom data ratio mixer in the DataFlex framework for dynamic domain ratios during training.

#### Step 1: Create Mixer Implementation File

First, create a new Python file in the specified project path to implement the core logic of the custom mixer.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/mixer/random_mixer.py`
2. **File Content**: Define a new class `RandomMixer` that inherits from `dataflex.train.mixer.base_mixer.Mixer`.

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
        Randomly generate a ratio vector.

        Returns:
            np.ndarray: Normalized ratio array with length equal to the number of sources.
        """
        k = len(self.mixture_manager.names)
        np.random.seed(self.seed)
        raw = np.random.random(k)
        probs = raw / raw.sum()  # Normalize
        logger.info(f"[RandomMixer] Step {step_id} Generated proportions: {probs}")

        return probs
```

**Key Points**:

- `@register_mixer('random')`: This decorator registers your `RandomMixer` class in the DataFlex framework with the unique name `random`. This name will be used in subsequent configuration files.
- `RandomMixer(Mixer)`: Your custom class must inherit from the framework's `Mixer` base class.
- `__init__`: Constructor for necessary initialization. Call `super().__init__(...)` to ensure base class initialization logic is properly executed.
- `mix`: Core method for implementing data ratio algorithm. You need to override this method according to your requirements, returning a normalized ratio array with length equal to the number of sources.

#### Step 2: Import New Module

To enable the DataFlex framework to recognize and load your newly created mixer, edit the `__init__.py` file in the same directory to expose your new module.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/mixer/__init__.py`
2. **Add Content**: Add the following line at the end of the file

```python
from .random_mixer import RandomMixer
```

#### Step 3: Configure Mixer Parameters

Finally, define your new mixer and its parameters in the YAML configuration file for convenient use in experiments.

1. **File Path**: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. **Add Configuration**: Add a new entry for your `random` mixer under the `mixers` configuration block.

```yaml
mixers:
  # ...
  random:
    name: random
    params:
      seed: 42
  # ...
```

**Key Points**:

- `params`: All parameters defined under this block will be passed as keyword arguments to the `__init__` constructor of the `RandomMixer` class. For example, the `seed` value here will be passed to the `seed` parameter of the `__init__` method.

## Weight Trainer Details

Weight Trainer allows you to dynamically adjust sample weights during backpropagation based on sample importance at specific training stages.

**Working Mechanism**: The external Weight Trainer (`weight_trainer.py`) calls the Weighter component's `training_step` method during the training loop to execute training. The `training_step` method is uniformly implemented by the base class `Weighter`, responsible for the complete process of forward propagation, loss calculation, and backpropagation. Specific weighting logic is completed by calling the `get_weighted_loss` method implemented by subclasses.

### Parameter Configuration

When using Weight Trainer, add the following DataFlex-specific parameters to your `.yaml` configuration file:

```yaml
train_type: dynamic_weight   # Select trainer type. Available options:
                          # "dynamic_select" - Dynamic selection trainer
                          # "dynamic_mix" - Dynamic mix trainer
                          # "dynamic_weight" - Dynamic weighting trainer
                          # "static" - Default static trainer
components_cfg_file: src/dataflex/configs/components.yaml
component_name: loss  # Select component name, corresponding to components defined in components_cfg_file
warmup_step: 1
train_step: 3 # Total training steps (including warmup)
```

### Parameter Details

- `train_type`: Defines training type. `dynamic_weight` enables Weight Trainer.
- `component_name`: Defines specific data weighting strategy. For example, `loss` uses loss-based weighter.
- `components_cfg_file`: Defines parameter file for the strategy, containing specific parameters for the corresponding strategy.
- `warmup_step`: Before the first dynamic weighting, the model needs `warmup_step` steps of regular training. This helps the model establish initial understanding of data distribution.
- `train_step`: Total training steps (including warmup). Weight Trainer will dynamically weight samples at each training step after warmup completion.

### How to Add Custom Weighter in DataFlex

This document uses `custom_weighter` as an example to detail how to add and configure a custom sample weighter in the DataFlex framework for dynamic sample weight adjustment during training.

#### Step 1: Create Weighter Implementation File

First, create a new Python file in the specified project path to implement the core logic of the custom weighter.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/weighter/custom_weighter.py`
2. **File Content**: Define a new class `CustomWeighter` that inherits from `dataflex.train.weighter.base_weighter.Weighter`.

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
        Custom weighter constructor
        
        Args:
            strategy: Weighting strategy, such as "uniform", "loss_based", etc.
            **kwargs: Additional parameters passed to base class
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
            losses: Per-sample loss for current device (B,)
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

**Key Points**:

- `@register_weighter('custom')`: This decorator registers your `CustomWeighter` class in the DataFlex framework with the unique name `custom`. This name will be used in subsequent configuration files.
- `CustomWeighter(Weighter)`: Your custom class must inherit from the framework's `Weighter` base class. The base class already implements the `training_step` method and `_per_sample_loss_from_logits` helper method.
- `__init__`: Constructor for necessary initialization. Call `super().__init__(**kwargs)` to ensure base class initialization logic is properly executed.
- `get_weighted_loss`: This is the core abstract method you need to implement to define the sample weighting algorithm. The base class's `training_step` method automatically calls this method to get weighted loss. The external Weight Trainer (`weight_trainer.py`) executes complete training steps including forward propagation, loss calculation, weighting, and backpropagation by calling the base class's `training_step` method.

#### Step 2: Import New Module

To enable the DataFlex framework to recognize and load your newly created weighter, edit the `__init__.py` file in the same directory to expose your new module.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/weighter/__init__.py`
2. **Add Content**: Add the following line at the end of the file

```python
from .custom_weighter import CustomWeighter
```

#### Step 3: Configure Weighter Parameters

Finally, define your new weighter and its parameters in the YAML configuration file for convenient use in experiments.

1. **File Path**: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. **Add Configuration**: Add a new entry for your `custom` weighter under the `weighters` configuration block.

```yaml
weighters:
  # ...
  custom:
    name: custom
    params:
      strategy: uniform
  # ...
```

**Key Points**:

- `params`: All parameters defined under this block will be passed as keyword arguments to the `__init__` constructor of the `CustomWeighter` class. For example, the `strategy` value here will be passed to the `strategy` parameter of the `__init__` method.

## Integration with LlamaFactory

DataFlex is fully compatible with LlamaFactory's configuration and usage:

1. **Configuration Compatibility**: Add DataFlex parameters on top of LlamaFactory configuration
2. **Command Consistency**: Use `dataflex-cli` instead of `llamafactory-cli`
3. **Feature Preservation**: Support all original LlamaFactory features
4. **Seamless Switching**: Can revert to original training mode with `train_type: static`

This design ensures users can progressively adopt DataFlex features without major modifications to existing workflows.