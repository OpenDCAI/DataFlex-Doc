---
title: 向Dataflex添加算子
createTime: 2025/06/30 19:19:16
permalink: /zh/guide/translation/
icon: basil:lightning-alt-outline
---

# 向Dataflex添加算子

# 如何在 DataFlex 中添加自定义选择器

本文档将以 `custom_selector` 为例，详细介绍如何在 DataFlex 框架中添加并配置一个自定义的数据选择器，实现训练过程中的动态数据点选择。

## 步骤一：创建选择器实现文件

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

### 关键点说明：

* `@register_selector('custom')`: 这是一个装饰器，用于将您的 `CustomSelector` 类注册到 DataFlex 框架中，并赋予其一个唯一的名称 `custom`。这个名称将在后续的配置文件中使用。
* `CustomSelector(Selector)`: 您的自定义类必须继承自框架提供的 `Selector` 基类。
* `__init__`: 构造函数用于执行必要的初始化操作。调用 `super().__init__(...)` 来确保基类的初始化逻辑被正确执行。
* `select`: 这是实现数据选择算法的核心方法。您需要根据自己的需求重写此方法。
* `warmup` (可选): 您还可以根据需要重写 `warmup` 方法，用于选择用于 warmup 的数据。默认随机采样数据用于 warmup 阶段训练。

## 步骤二：导入新模块

为了让 DataFlex 框架能够识别并加载您新创建的选择器，需要编辑该目录下的 `__init__.py` 文件，以暴露您的新模块。

1. **文件路径**: `DataFlex-Preview/src/dataflex/train/selector/__init__.py`
2. **添加内容**: 在文件末尾添加以下行，以导入 `CustomSelector` 类。

```python
from .custom_selector import *
```

## 步骤三：配置选择器参数

最后，在 YAML 配置文件中定义您的新选择器及其参数，以便在实验中方便地调用。

1. **文件路径**: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. **添加配置**: 在 `selectors` 配置块下，为您的 `custom` 选择器添加新的条目。

```yaml
selectors:
  ...
  # 添加您的自定义选择器配置
  custom:
    name: custom
    params:
      cache_dir: ../dataflex_saves/custom_output
  ...
```

### 关键点说明：

* `params::` 该块下定义的所有参数都将作为关键字参数传递给 `CustomSelector` 类的 `__init__` 构造函数。例如，这里的 `cache_dir` 值会传递给 `__init__` 方法的 `cache_dir` 参数。

```

这个 Markdown 格式的文档可以直接用于文档生成、GitHub README 等场景。
```
