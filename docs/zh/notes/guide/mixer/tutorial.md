---
title: 向Dataflex添加Mixer
createTime: 2025/06/30 19:19:16
permalink: /zh/guide/mixer/tutorial/
icon: solar:add-circle-outline
---

# 向Dataflex添加Mixer

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

## 如何在 DataFlex 中添加自定义 Mixer

本文档将以 `random_mixer` 为例，详细介绍如何在 DataFlex 框架中添加并配置一个自定义的数据配比器，实现训练过程中的动态领域配比。

### 步骤一：创建配比器实现文件

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

#### 关键点说明

- `@register_mixer('random')`: 这是一个装饰器，用于将您的 `RandomMixer` 类注册到 DataFlex 框架中，并赋予其一个唯一的名称 `random`。这个名称将在后续的配置文件中使用。
- `RandomMixer(Mixer)`: 您的自定义类必须继承自框架提供的 `Mixer` 基类。
- `__init__`: 构造函数用于执行必要的初始化操作。调用 `super().__init__(...)` 来确保基类的初始化逻辑被正确执行。
- `mix`: 这是实现数据配比算法的核心方法。您需要根据自己的需求重写此方法，需要返回长度为源数量的归一化比例数组。

### 步骤二：导入新模块

为了让 DataFlex 框架能够识别并加载您新创建的配比器，需要编辑该目录下的 `__init__.py` 文件，以暴露您的新模块。

1. **文件路径**: `DataFlex-Preview/src/dataflex/train/mixer/__init__.py`
2. **添加内容**: 在文件末尾添加以下行

```python
from .random_mixer import RandomMixer
```

### 步骤三：配置配比器参数

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

#### 关键点说明

- `params`: 该块下定义的所有参数都将作为关键字参数传递给 `RandomMixer` 类的 `__init__` 构造函数。例如，这里的 `seed` 值会传递给 `__init__` 方法的 `seed` 参数。