---
title: DoReMi 数据混合器
createTime: 2025/01/30 10:00:00
icon: material-symbols:balance
permalink: /zh/guide/mixer/doremi/
---

# DoReMi 数据混合器

DoReMi (Domain Reweighting with Minimax Optimization) 是一种用于优化多领域数据混合比例的算法。通过在小型代理模型上进行领域权重优化，可以找到适用于大规模模型训练的最优数据混合策略。

## 算法概述

DoReMi 算法分为三个步骤：

1. **Step 1**: 使用参考权重训练参考模型（Reference Model）
2. **Step 2**: 在小型代理模型上动态优化领域权重（Proxy Model）
3. **Step 3**: 使用优化后的权重训练大规模目标模型

## 三步训练流程

### Step 1: 参考模型训练

使用初始的领域权重（通常是均匀分布或经验权重）训练一个参考模型。这个模型将作为后续权重优化的基准。

**配置文件**: `doremi_step1_static_qwen_pt_full.yaml`

```yaml
### dynamic_train - DoReMi Step 1: Reference Model Training
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: static  # 使用静态混合器
mixture_sample_rule: mixture
init_mixture_proportions: [0.5, 0.5]  # 初始权重，这里使用均匀分布
static_mix: true
warmup_step: 100
update_step: 200
update_times: 3
```

**关键参数说明**:
- `component_name: static`: 使用静态混合器，在整个训练过程中保持权重不变
- `init_mixture_proportions`: 初始领域权重，需与数据集数量对应
- `static_mix: true`: 启用静态混合模式

**在 components.yaml 中的配置**:

```yaml
mixers:
  static:
    name: static
    params:
      proportions: [0.5, 0.5]  # 对应各个域的比例
      # proportions: null  # 使用均匀分布
```

### Step 2: 代理模型权重优化

使用 DoReMi 算法在小型代理模型上动态优化领域权重。算法会通过计算各领域的过剩损失（excess loss）来调整权重。

**配置文件**: `doremi_step2_dynamic_qwen_pt_full.yaml`

```yaml
### dynamic_train - DoReMi Step 2: Proxy Model Training
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: doremi  # 使用 DoReMi 混合器
mixture_sample_rule: mixture
init_mixture_proportions: [0.5, 0.5]  # 初始权重
warmup_step: 100  # 预热步数
update_step: 200  # 权重更新间隔
update_times: 3   # 权重更新次数
```

**在 components.yaml 中的配置**:

```yaml
mixers:
  doremi:
    name: doremi
    params:
      # Step 1 训练得到的参考模型路径
      reference_model_path: /path/to/doremi_step1_result/checkpoint-xxx
      # 权重更新学习率 (DoReMi 论文中的 eta)
      reweight_eta: 1.0
      # 权重平滑参数 (DoReMi 论文中的 epsilon)
      reweight_eps: 1e-3
      # 每个领域评估的样本数
      num_eval_samples: 1000
      # 评估时的批次大小
      eval_batch_size: 8
```

**关键参数说明**:
- `reference_model_path`: Step 1 训练得到的参考模型检查点路径
- `reweight_eta`: 权重更新的学习率，控制权重调整幅度
- `reweight_eps`: 平滑参数，防止某些领域权重过小
- `num_eval_samples`: 每个领域用于计算过剩损失的样本数
- `warmup_step`: 在开始权重优化前的预热训练步数
- `update_step`: 每隔多少步更新一次领域权重

**权重日志**:

训练过程中会自动生成 `doremi_weights.jsonl` 文件，记录每次权重更新的详细信息：

```json
{"step": 100, "timestamp": "2025-01-30 10:00:00", "domain_names": ["wiki", "c4"], "domain_weights": [0.3, 0.7], "perdomain_scores": [2.5, 3.2], "reweight_eta": 1.0, "reweight_eps": 0.001}
{"step": 300, "timestamp": "2025-01-30 10:10:00", "domain_names": ["wiki", "c4"], "domain_weights": [0.25, 0.75], "perdomain_scores": [2.3, 3.5], "reweight_eta": 1.0, "reweight_eps": 0.001}
```

### Step 3: 目标模型训练

使用 Step 2 优化得到的最终权重，训练大规模目标模型。

**配置文件**: `doremi_step3_static_qwen_pt_full.yaml`

```yaml
### dynamic_train - DoReMi Step 3: Large Model Training with Optimized Weights
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: static  # 使用静态混合器
mixture_sample_rule: mixture
init_mixture_proportions: [0.3, 0.7]  # 使用 Step 2 优化得到的最终权重
static_mix: true
warmup_step: 100
update_step: 200
update_times: 3
```

**关键步骤**:
1. 从 Step 2 的 `doremi_weights.jsonl` 文件中提取最终的优化权重
2. 将权重填入 `init_mixture_proportions` 配置项
3. 使用静态混合器进行训练

## 完整训练示例

```bash
# Step 1: 训练参考模型
llamafactory-cli train examples/train_full/mixers/doremi_step1_static_qwen_pt_full.yaml

# Step 2: 优化领域权重（在小型代理模型上）
# 注意：需要先修改 components.yaml 中的 reference_model_path
llamafactory-cli train examples/train_full/mixers/doremi_step2_dynamic_qwen_pt_full.yaml

# Step 3: 使用优化权重训练目标模型
# 注意：需要将 Step 2 的最终权重填入配置文件
llamafactory-cli train examples/train_full/mixers/doremi_step3_static_qwen_pt_full.yaml
```

## 权重提取和分析

从 Step 2 的输出目录中读取优化后的权重：

```python
import json

# 读取权重日志
weights_history = []
with open('doremi_step2_result/doremi_weights.jsonl', 'r') as f:
    for line in f:
        weights_history.append(json.loads(line))

# 获取最终权重
final_weights = weights_history[-1]['domain_weights']
domain_names = weights_history[-1]['domain_names']

print("优化后的领域权重:")
for name, weight in zip(domain_names, final_weights):
    print(f"  {name}: {weight:.4f}")

# 可视化权重变化趋势
import matplotlib.pyplot as plt
import numpy as np

steps = [entry['step'] for entry in weights_history]
weights_matrix = np.array([entry['domain_weights'] for entry in weights_history])

plt.figure(figsize=(10, 6))
for i, name in enumerate(domain_names):
    plt.plot(steps, weights_matrix[:, i], label=name, marker='o')
plt.xlabel('Training Step')
plt.ylabel('Domain Weight')
plt.title('DoReMi Domain Weight Evolution')
plt.legend()
plt.grid(True)
plt.savefig('doremi_weights_evolution.png')
plt.show()
```

## 最佳实践

### 1. 参考模型训练

- 使用均匀分布或基于数据集大小的比例作为初始权重
- 确保参考模型充分收敛，建议训练至少一个完整 epoch
- 保存多个检查点，选择验证损失最低的模型

### 2. 权重优化

- 代理模型建议使用小型模型（如 0.5B-1B 参数）以降低计算成本
- `num_eval_samples` 设置在 1000-5000 之间，平衡评估准确性和速度
- `reweight_eta` 通常设置为 1.0，可根据收敛情况调整
- 建议至少进行 3-5 次权重更新（`update_times`）以观察收敛趋势

### 3. 目标模型训练

- 使用 Step 2 最后一次更新的权重，而非中间结果
- 可以对比使用优化权重和均匀分布的训练效果
- 建议在下游任务上评估模型性能

## 常见问题

### Q: 为什么需要三个步骤？

A: DoReMi 的核心思想是通过对比参考模型和代理模型的损失来优化权重。Step 1 提供基准，Step 2 在小模型上快速找到最优权重，Step 3 将结果应用到大模型训练。

### Q: 权重如何更新？

A: 使用指数梯度上升算法（Exponentiated Gradient Ascent）。领域的过剩损失越高，其权重会增加；反之则减少。具体公式：

$$
w_i^{(t+1)} \propto w_i^{(t)} \cdot \exp(\eta \cdot \text{excess\_loss}_i^{(t)})
$$

### Q: 如何选择初始权重？

A: 可以选择：
- 均匀分布：`[1/k, 1/k, ..., 1/k]`
- 基于数据集大小的比例
- 基于领域先验知识的比例

### Q: 没有参考模型可以运行吗？

A: 可以。如果 `reference_model_path` 设置为 `null`，算法会直接使用代理模型的损失值进行优化（相当于最小化训练损失），但需要注意的是这已经不属于DoReMi的算法部分，因此仅推荐作为debug阶段使用该参数缺省。

## 参考资料

- 论文: [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429)
- 项目地址: [DataFlex GitHub](https://github.com/OpenDCAI/DataFlex)

## 相关组件

- [静态混合器 (Static Mixer)](/zh/guide/mixer/static/)
- [数据混合管理器 (Mixture Manager)](/zh/guide/data/mixture/)

