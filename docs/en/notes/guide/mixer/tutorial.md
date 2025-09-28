---
title: Add Mixer to Dataflex
createTime: 2025/06/30 19:19:16
permalink: /en/guide/mixer/tutorial/
icon: solar:add-circle-outline
---

# Add Mixer to Dataflex

## Mix Trainer Overview

Mix Trainer allows you to dynamically adjust the domain data proportions for subsequent training based on the model's current state at specific stages of training.

### Parameter Configuration

When using Mix Trainer, you need to add the following DataFlex-specific parameters to your `.yaml` configuration file:

```yaml
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: random
mixture_sample_rule: mixture     # Initial sampling rule: mixture uses init_mixture_proportions (adjustable),
                                 # stratified uses fixed proportions by source dataset size, uniform uses fixed uniform distribution
init_mixture_proportions: [0.7, 0.3]  # Initial proportions, required when mixture_sample_rule='mixture'
warmup_step: 4
update_step: 3
update_times: 2
```

### Parameter Details

- `train_type`: Defines the training type. `dynamic_mix` enables Mix Trainer.
- `component_name`: Defines the specific strategy for data mixing. For example, `random` uses a random domain mixer.
- `components_cfg_file`: Defines the parameter file containing specific parameters for the corresponding strategy.
- `mixture_sample_rule`: Initial sampling rule, required. `mixture` uses proportions from `init_mixture_proportions` (adjustable), `stratified` uses fixed proportions by source dataset size, `uniform` uses fixed uniform distribution.
- `init_mixture_proportions`: Initial sampling proportions, required when `mixture_sample_rule='mixture'`.
- `warmup_step`: Before the first dynamic proportion update, the model needs to perform `warmup_step` steps of regular training. This helps the model establish initial understanding of data distribution.
- `update_step`: Frequency of domain proportion updates. After every `update_step` training steps, the Mixer will be triggered to update domain proportions for the next training phase.
- `update_times`: Total number of dynamic data proportion calculations during the entire training process. Therefore, total training steps = `(update_times * update_step + warmup_step) * global_batch_size`

### Static Mixing Configuration

Mix Trainer supports static mixing mode by setting `static_mix: true` to fix initial proportions:

```yaml
train_type: dynamic_mix
static_mix: true                      # Whether to fix initial static mixing proportions (only effective in dynamic_mix trainer)
mixture_sample_rule: mixture          # Initial sampling rule
init_mixture_proportions: [0.7, 0.3]  # Initial proportions, can be adjusted by additional algorithms
train_step: 3                         # Total training steps (only effective in dynamic_mix trainer), excluding warmup and update steps
```

When static mixing is enabled, the training process will use fixed `init_mixture_proportions` without dynamic adjustment.

## How to Add Custom Mixer in DataFlex

This document will use `random_mixer` as an example to detail how to add and configure a custom data mixer in the DataFlex framework for dynamic domain proportion adjustment during training.

### Step 1: Create Mixer Implementation File

First, create a new Python file in the specified project path to implement the core logic of your custom mixer.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/mixer/random_mixer.py`
2. **File Content**: In this file, define a new class `RandomMixer` that inherits from `dataflex.train.mixer.base_mixer.Mixer`.

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
        Randomly generate a set of proportion vectors.

        Returns:
            np.ndarray: Normalized proportion array with length equal to the number of sources.
        """
        k = len(self.mixture_manager.names)
        np.random.seed(self.seed)
        raw = np.random.random(k)
        probs = raw / raw.sum()  # Normalize
        logger.info(f"[RandomMixer] Step {step_id} Generated proportions: {probs}")

        return probs
```

#### Key Points

- `@register_mixer('random')`: This decorator registers your `RandomMixer` class into the DataFlex framework with the unique name `random`. This name will be used in configuration files.
- `RandomMixer(Mixer)`: Your custom class must inherit from the framework's `Mixer` base class.
- `__init__`: Constructor for necessary initialization. Call `super().__init__(...)` to ensure base class initialization is properly executed.
- `mix`: Core method implementing the data mixing algorithm. You need to override this method according to your needs, returning a normalized proportion array with length equal to the number of sources.

### Step 2: Import New Module

To enable DataFlex framework to recognize and load your newly created mixer, edit the `__init__.py` file in the directory to expose your new module.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/mixer/__init__.py`
2. **Add Content**: Add the following line at the end of the file

```python
from .random_mixer import RandomMixer
```

### Step 3: Configure Mixer Parameters

Finally, define your new mixer and its parameters in the YAML configuration file for convenient use in experiments.

1. **File Path**: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. **Add Configuration**: Under the `mixers` configuration block, add a new entry for your `random` mixer.

```yaml
mixers:
  # ...
  random:
    name: random
    params:
      seed: 42
  # ...
```

#### Key Points

- `params`: All parameters defined under this block will be passed as keyword arguments to the `__init__` constructor of the `RandomMixer` class. For example, the `seed` value here will be passed to the `seed` parameter of the `__init__` method.