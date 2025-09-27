---
title: Contribute to Dataflex Selector
createTime: 2025/06/30 19:19:16
permalink: /en/guide/translation/
icon: basil:lightning-alt-outline
---

# Less Algorithm

This document will detail how to add and configure a custom data selector in the DataFlex framework, enabling dynamic sample selection during the training process, using `custom_selector` as an example.
## Step 1: Create the Selector Implementation File

First, create a new Python file in the specified project path to implement the core logic of your custom selector.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/selector/custom_selector.py`
2. **File Content**: In this file, define a new class `CustomSelector` that inherits from `dataflex.train.selector.base_selector.Selector`.

```python
from dataflex.core.registry import register_selector
from .base_selector import logger, Selector

@register_selector('custom')
class CustomSelector(Selector):
    """
    An example implementation of a custom data selector.
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
        The core selection logic.
        This method defines how to select samples from the dataset.

        Args:
            model: The current model.
            step_id (int): The current training step.
            num_samples (int): The number of samples to select.

        Returns:
            list: A list of indices of the selected samples.
        """
        # Example logic: simply return a list of indices from 0 to num_samples-1.
        # You can implement more complex selection algorithms here.
        return list(range(num_samples))
```

### Key Points Explanation:

* `@register_selector('custom')`: This decorator registers your `CustomSelector` class into the DataFlex framework and assigns it a unique name, `custom`. This name will be used in configuration files later.
* `CustomSelector(Selector)`: Your custom class must inherit from the `Selector` base class provided by the framework.
* `__init__`: The constructor is used to perform necessary initialization tasks. It calls `super().__init__(...)` to ensure that the base class initialization logic is executed correctly.
* `select`: This is the core method where you implement your data selection algorithm. You should override this method according to your needs.
* `warmup` (optional): You can also override the `warmup` method if you need to select data for the warmup phase of training. By default, data is randomly sampled during the warmup phase.

## Step 2: Import the New Module

In order for DataFlex to recognize and load your newly created selector, you need to edit the `__init__.py` file in this directory to expose your new module.

1. **File Path**: `DataFlex-Preview/src/dataflex/train/selector/__init__.py`
2. **Add Content**: Add the following line at the end of the file to import the `CustomSelector` class.

```python
from .custom_selector import *
```

## Step 3: Configure the Selector Parameters

Finally, define your new selector and its parameters in a YAML configuration file so it can be easily called during experiments.

1. **File Path**: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. **Add Configuration**: Under the `selectors` configuration block, add a new entry for your `custom` selector.

```yaml
selectors:
  ...
  # Add your custom selector configuration
  custom:
    name: custom
    params:
      cache_dir: ../dataflex_saves/custom_output
  ...
```

### Key Points Explanation:

* `params::` All parameters defined under this block will be passed as keyword arguments to the `__init__` constructor of the `CustomSelector` class. For example, the value of `cache_dir` here will be passed to the `cache_dir` parameter of the `__init__` method.

```

This English version of the tutorial can be used directly for documentation purposes, README files, or other resources where an English version is required.
```
