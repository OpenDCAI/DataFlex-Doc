# Overview

DataFlex 是一个基于 LlamaFactory 的高级动态训练框架。它通过在训练过程中智能地调度数据，支持动态指定样本的训练顺序和领域配比和动态权重，旨在提升模型训练的效率与最终效果。其设计思想是与 LlamaFactory 无缝集成，为研究者和开发者提供更灵活、更强大的训练控制能力。

# Usage

## Installation:

通过以下命令以可编辑模式安装 DataFlex：

```Plain
git clone https://github.com/OpenDCAI/DataFlex-Preview.git
cd DataFlex-Preview
pip install -e .
```
同时环境中需要安装llamafactory

## Example

启动训练的命令与 LlamaFactory 非常相似。以下是一个使用 `LESS` 的示例，具体原理参考论文https://arxiv.org/abs/2402.04333

```Plain
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/less.yaml
```

​**注意**​：与标准 LlamaFactory 不同，您的 `.yaml` 配置文件中除了需要包含 LlamaFactory 的标准训练参数外，还必须指定 DataFlex 的特定参数。

# Core concepts: Trainer and Selector/Mixer/Weighter

DataFlex 提供了三大核心训练器，它们都可以无缝接入 LlamaFactory 的训练流程：

* ​**Select Trainer (动态选择训练器)**​: 在训练过程中，根据预设策略（Selector）动态地从数据集中挑选出一部分样本用于接下来的训练，例如优先训练模型认为“难”的样本。
* ​**Mix Trainer (动态配比训练器)**​:  支持在训练中动态调整不同来源或领域数据的混合比例。
* ​**Weight Trainer (动态加权训练器)**​:  支持在训练中动态调整样本反向传播时的权重，增大模型偏好的数据的学习力度。

## Select Trainer详解

Select Trainer 允许您在训练的特定阶段，根据模型的当前状态动态调整后续的训练数据顺序。

### 参数配置

当使用 Select Trainer 时，需要在 `.yaml` 配置文件中添加以下 `dataflex` 特定参数，并指定：

```Plain
# select_trainer paras 
train_type: dynamic_select   # 必须，指定使用 Select Trainer
                             # 从[dynamic_select, dynamic_mix, dynamic_weight, static]中选择
                             # 分别对应动态选择、动态混合、动态加权和静态（即llamafactory原始训练流程）
component_name: loss         # 必须，指定具体使用的选择器策略，如 loss 或 less
components_cfg_file: src/dataflex/configs/components.yaml    # 必须，指定具体使用的选择器的参数
warmup_step: 200             # 必选，在首次数据选择前，模型预热训练的步数
update_step: 1000            # 必须，每隔多少步（step）执行一次数据选择
update_times: 2              # 必须，数据选择执行的总次数
```

**参数详解:**

* `train_type`: 定义训练类型。`dynamic_select` 表示启用 Select Trainer。
* `component_name`: 定义数据选择的具体策略。例如，`loss` 表示使用基于损失值的选择器。
* `components_cfg_file`: 定义数据选择策略的参数文件，包含对应策略的特定参数。
* `warmup_step`: 在执行第一次动态选择之前，模型需要先进行 `warmup_step` 步的常规训练。这有助于模型建立对数据分布的初步认知。
* `update_step`: 数据选择的频率。每当训练进行 `update_step` 步后，Selector 将被触发，重新选择 `update_step * global_batch_size` 个样本用于下一阶段的训练。
* `update_times`: 整个训练过程中，动态数据选择执行的总次数。因此总的训练步数为`(update_times * update_step + warmup_step) * global_batch_size`

### **如何在 DataFlex 中添加自定义选择器 ​**

本文档将以 `custom_selector` 为例，详细介绍如何在 DataFlex 框架中添加并配置一个自定义的数据选择器，实现训练过程中的动态数据点选择。

#### 步骤一：创建选择器实现文件

首先，在项目指定路径下创建一个新的 Python 文件，用于实现自定义选择器的核心逻辑。

1. ​**文件路径**​: `DataFlex-Preview/src/dataflex/train/selector/custom_selector.py`
2. ​**文件内容**​: 在该文件中，定义一个继承自 `dataflex.train.selector.base_selector.Selector` 的新类 `CustomSelector`。

```Python
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

​**关键点说明**​:

* `@register_selector('custom')`: 这是一个装饰器，用于将您的 `CustomSelector` 类注册到 DataFlex 框架中，并赋予其一个唯一的名称 `custom`。这个名称将在后续的配置文件中使用。
* `CustomSelector(Selector)`: 您的自定义类必须继承自框架提供的 `Selector` 基类。
* `init`: 构造函数用于执行必要的初始化操作。调用 `super().__init__(...)` 来确保基类的初始化逻辑被正确执行。
* `select`: 这是实现数据选择算法的核心方法。您需要根据自己的需求重写此方法。
* `warmup` (可选): 您还可以根据需要重写 `warmup` 方法，用于选择用于warmup的数据。默认随机采样数据用于warmup阶段训练。

#### **步骤二：导入新模块**

为了让 DataFlex 框架能够识别并加载您新创建的选择器，需要编辑该目录下的 `init.py` 文件，以暴露您的新模块。

1. ​**文件路径**​: `DataFlex-Preview/src/dataflex/train/selector/__init__.py`
2. ​**添加内容**​: 在文件末尾添加以下行，以导入 `CustomSelector` 类。

```Python
from .custom_selector import *
```

#### **步骤三：配置选择器参数**

最后，在 YAML 配置文件中定义您的新选择器及其参数，以便在实验中方便地调用。

1. ​**文件路径**​: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. ​**添加配置**​: 在 `selectors` 配置块下，为您的 `custom` 选择器添加新的条目。

```YAML
selectors:
  ...
  # 添加您的自定义选择器配置
  custom:
    name: custom
    params:
      cache_dir: ../dataflex_saves/custom_output
  ...
```

​**关键点说明**​:

* `params:`: 该块下定义的所有参数都将作为关键字参数传递给 `CustomSelector` 类的 `init` 构造函数。例如，这里的 `cache_dir` 值会传递给 `init` 方法的 `cache_dir` 参数。

## Mix Trainer详解

Mix Trainer 允许您在训练的特定阶段，根据模型的当前状态动态调整后续的领域数据配比

### 参数配置

当使用 Mix Trainer 时，需要在 `.yaml` 配置文件中添加以下 `dataflex` 特定参数，并指定：

```Plain
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: random
mixture_sample_rule: mixture     # 初始采样规则，mixture为根据init_mixture_proportions比例混合（可动态调整），stratified为固定按源数据集大小比例分层，uniform为固定均匀分布
init_mixture_proportions: [0.7, 0.3]         # 对应初始的比例，如果mixture_sample_rule为mixture必须设置
warmup_step: 4
update_step: 3
update_times: 2
```

**参数详解:**

* `train_type`: 定义训练类型。dynamic\_mix 表示启用 Mix Trainer。
* `component_name`: 定义数据选择的具体策略。例如，random 表示使用随机的领域配比器。
* `components_cfg_file`: 定义策略的参数文件，包含对应策略的特定参数。
* `mixture_sample_rule`：初始采样规则，必选，mixture为根据init\_mixture\_proportions比例混合（可动态调整），stratified为固定按源数据集大小比例分层，uniform为固定均匀分布
* `init_mixture_proportions`：初始采样对应的对应的比例，`mixture_sample_rule='mixture'`时需要指定
* `warmup_step`: 在执行第一次动态配比更新前，模型需要先进行 `warmup_step` 步的常规训练。这有助于模型建立对数据分布的初步认知。
* `update_step`: 领域配比更新的频率。每当训练进行 `update_step` 步后，Mixer 将被触发，更新领域配比用于下一阶段的训练。
* `update_times`: 整个训练过程中，动态数据配比计算的总次数。因此总的训练步数为`(update_times * update_step + warmup_step) * global_batch_size`

### **如何在 DataFlex 中添加自定义Mixer**

本文档将以 `random_mixer` 为例，详细介绍如何在 DataFlex 框架中添加并配置一个自定义的数据配比器，实现训练过程中的动态的领域配比。

#### 步骤一：创建选择器实现文件

首先，在项目指定路径下创建一个新的 Python 文件，用于实现自定义选择器的核心逻辑。

1. ​**文件路径**​: `DataFlex-Preview/src/dataflex/train/mixer/random_selector.py`
2. ​**文件内容**​: 在该文件中，定义一个继承自 `dataflex.train.mixer.base_mixer.Mixer` 的新类 `RandomMixer`。

```Python
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

​**关键点说明**​:

* `@register_selector('random')`: 这是一个装饰器，用于将您的 `RandomMixer` 类注册到 DataFlex 框架中，并赋予其一个唯一的名称 `random`。这个名称将在后续的配置文件中使用。
* `RandomMixer(Mixer)`: 您的自定义类必须继承自框架提供的 `Mixer` 基类。
* `init`: 构造函数用于执行必要的初始化操作。调用 `super().__init__(...)` 来确保基类的初始化逻辑被正确执行。
* `mix`: 这是实现数据选择算法的核心方法。您需要根据自己的需求重写此方法，需要返回长度为源数量的归一化比例数

#### **步骤二：导入新模块**

为了让 DataFlex 框架能够识别并加载您新创建的选择器，需要编辑该目录下的 `init.py` 文件，以暴露您的新模块。

1. ​**文件路径**​: `DataFlex-Preview/src/dataflex/train/mixer/__init__.py`
2. ​**添加内容**​: 在文件末尾添加以下行

```Python
from .random_mixer import RandomMixer
```

#### **步骤三：配置选择器参数**

最后，在 YAML 配置文件中定义您的新选择器及其参数，以便在实验中方便地调用。

1. ​**文件路径**​: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. ​**添加配置**​: 在 `selectors` 配置块下，为您的 `custom` 选择器添加新的条目。

```YAML
mixers:
  ...
  random:
    name: random
    params:
      seed: 42
  ...
```

​**关键点说明**​:

* `params:`: 该块下定义的所有参数都将作为关键字参数传递给 `RandomMixer` 类的 `init` 构造函数。例如，这里的 `seed` 值会传递给 `init` 方法的 `seed` 参数。

## Weight Trainer详解

Weight Trainer 允许您在训练的特定阶段，根据样本的重要性动态调整样本在反向传播时的权重

### 参数配置

当使用 Weight Trainer 时，需要在 `.yaml` 配置文件中添加以下 `dataflex` 特定参数，并指定：

```Plain
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

**参数详解:**

* `train_type`: 定义训练类型。`dynamic_weight` 表示启用 Weight Trainer。
* `component_name`: 定义数据加权的具体策略。例如，`loss` 表示使用基于损失值的加权器。
* `components_cfg_file`: 定义策略的参数文件，包含对应策略的特定参数。
* `warmup_step`: 在执行第一次动态加权前，模型需要先进行 `warmup_step` 步的常规训练。这有助于模型建立对数据分布的初步认知。
* `train_step`: 总训练步数（包括warmup），Weight Trainer 将在 warmup 完成后的每个训练步骤中对样本进行动态加权。

### **如何在 DataFlex 中添加自定义Weighter**

本文档将以 `loss_weighter` 为例，详细介绍如何在 DataFlex 框架中添加并配置一个自定义的样本加权器，实现训练过程中的动态样本权重调整。

#### 步骤一：创建加权器实现文件

首先，在项目指定路径下创建一个新的 Python 文件，用于实现自定义加权器的核心逻辑。

1. ​**文件路径**​: `DataFlex-Preview/src/dataflex/train/weighter/custom_weighter.py`
2. ​**文件内容**​: 在该文件中，定义一个继承或实现 `Weighter` 接口的新类 `CustomWeighter`。

```Python
from dataflex.core.registry import register_weighter
from dataflex.utils.logging import logger
from typing import Any, Union
from torch import nn
import torch

@register_weighter("custom")
class CustomWeighter:
    def __init__(self, strategy: str = "uniform"):
        """
        自定义加权器的构造函数
        
        Args:
            strategy: 加权策略，如 "uniform"、"loss_based" 等
        """
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
    
    def training_step(self, ctx, model, inputs, num_items_in_batch=None, use_weighter=False):
        """
        执行训练步骤，包含前向传播、损失计算、加权和反向传播
        
        Args:
            ctx: Trainer 上下文
            model: 模型
            inputs: 输入数据
            num_items_in_batch: 批次中的样本数量
            use_weighter: 是否使用加权器
            
        Returns:
            本步骤的损失值
        """
        model.train()
        inputs = ctx._prepare_inputs(inputs)
        
        with ctx.compute_loss_context_manager():
            loss, outputs = ctx.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch, return_outputs=True
            )
        
        if use_weighter:
            # 实现加权逻辑
            # 这里需要根据具体情况获取 per-sample 损失
            # 然后调用 get_weighted_loss 进行加权
            pass
        
        # 执行反向传播
        loss = loss / ctx.args.gradient_accumulation_steps
        ctx.accelerator.backward(loss)
        
        return loss.detach()
```

​**关键点说明**​:

* `@register_weighter('custom')`: 这是一个装饰器，用于将您的 `CustomWeighter` 类注册到 DataFlex 框架中，并赋予其一个唯一的名称 `custom`。这个名称将在后续的配置文件中使用。
* `get_weighted_loss`: 这是实现样本加权算法的核心方法。您需要根据自己的需求重写此方法，输入 per-sample 损失，返回加权后的总损失。
* `training_step`: 这是执行完整训练步骤的方法，包含前向传播、损失计算、加权处理和反向传播。在 warmup 阶段会跳过加权逻辑。

#### **步骤二：导入新模块**

为了让 DataFlex 框架能够识别并加载您新创建的加权器，需要编辑该目录下的 `init.py` 文件，以暴露您的新模块。

1. ​**文件路径**​: `DataFlex-Preview/src/dataflex/train/weighter/__init__.py`
2. ​**添加内容**​: 在文件末尾添加以下行

```Python
from .custom_weighter import CustomWeighter
```

#### **步骤三：配置加权器参数**

最后，在 YAML 配置文件中定义您的新加权器及其参数，以便在实验中方便地调用。

1. ​**文件路径**​: `DataFlex-Preview/src/dataflex/configs/components.yaml`
2. ​**添加配置**​: 在 `weighters` 配置块下，为您的 `custom` 加权器添加新的条目。

```YAML
weighters:
  ...
  custom:
    name: custom
    params:
      strategy: uniform
  ...
```

​**关键点说明**​:

* `params:`: 该块下定义的所有参数都将作为关键字参数传递给 `CustomWeighter` 类的 `init` 构造函数。例如，这里的 `strategy` 值会传递给 `init` 方法的 `strategy` 参数。
