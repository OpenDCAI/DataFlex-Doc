---
title: ODM 数据混合器
createTime: 2025/01/27 10:00:00
icon: material-symbols:casino
permalink: /zh/guide/mixer/odm/
---

# ODM 数据混合器

ODM (Online Data Mixing) 是一种在训练过程中动态优化多领域数据混合比例的算法。它使用多臂老虎机（Multi-Armed Bandits）中的 Exp3 算法，基于训练损失作为奖励信号自适应地调整领域权重。与 DoReMi 不同，ODM 不需要参考模型，因此更加高效且易于部署。

## 算法概述

ODM 使用 Exp3（Exploration and Exploitation 的指数权重）算法在训练过程中动态调整领域权重：

1. **预热阶段**：使用初始比例（均匀分布或指定值）进行固定步数的训练
2. **评估阶段**：通过采样批次评估每个领域的当前损失
3. **奖励更新**：使用重要性加权和移动平均更新累积估计奖励
4. **策略更新**：使用具有衰减探索率的 Exp3 算法更新采样策略

算法通过随时间衰减的探索率自动平衡探索（尝试所有领域）和利用（关注高奖励领域）。

## 核心特性

- **无需参考模型**：与 DoReMi 不同，ODM 直接使用训练模型工作
- **在线学习**：在训练过程中持续适应权重
- **重要性加权**：使用重要性加权奖励来处理非均匀采样
- **衰减探索**：探索率随时间衰减，以实现更好的收敛

## 配置说明

### 基础配置

**配置文件**: `odm_dynamic_qwen_pt_full.yaml`

```yaml
### dynamic_train - ODM: Online Data Mixing with Multi-Armed Bandits
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: odm  # 使用 ODM 混合器
mixture_sample_rule: mixture
init_mixture_proportions: [0.5, 0.5]  # 初始权重
warmup_step: 2000  # ODM 开始前的预热步数
update_step: 500   # 权重更新频率
update_times: -1   # -1 表示持续更新直到训练结束
```

**在 components.yaml 中的配置**:

```yaml
mixers:
  odm:
    name: odm
    params:
      # 指数移动平均的平滑参数 (0 到 1)
      alpha: 0.9
      
      # 使用初始比例的预热步数
      warmup_steps: 2000
      
      # 计算奖励时每个领域评估的样本数
      num_eval_samples: 500
      
      # 评估时的批次大小
      eval_batch_size: 8
      
      # 预热期的初始比例
      initial_proportions: [0.5, 0.5]
      # initial_proportions: null  # 使用均匀分布
```

### 训练配置参数

在训练配置文件中：

- **`warmup_step`**: 开始 ODM 前的步数（应与 components.yaml 中的 `warmup_steps` 匹配或更大）
- **`update_step`**: 权重更新的频率（每 N 步）
- **`update_times`**: 权重更新次数。使用 `-1` 表示持续更新直到训练结束
- **`train_step`**: 可选的显式总训练步数（覆盖 `num_train_epochs`）

## 训练流程

### 单步训练

与 DoReMi 的三步流程不同，ODM 只需要一次训练运行：

```bash
# 使用 ODM 的单一训练运行
llamafactory-cli train examples/train_full/mixers/odm_dynamic_qwen_pt_full.yaml
```

### 权重更新过程

在训练过程中，ODM 在每次更新时执行以下步骤：

1. **领域评估**：从每个领域采样并评估批次以计算当前损失
2. **奖励计算**：将损失转换为奖励（reward = loss / 10.0 以防止爆炸）
3. **重要性加权**：使用重要性加权更新累积估计奖励：`R̂_i += reward_i / π_i`
4. **策略更新**：使用 Exp3 算法更新领域权重：
   - 计算探索率：`ε_t = min{1/K, sqrt(ln(K) / (K * t))}`
   - 更新权重：`w_i = exp(ε_{t-1} * R̂_i) * scaling_factor + ε_t`
   - 归一化得到概率：`π_i = w_i / Σ_j w_j`

### 权重日志

训练过程中会自动生成 `odm_weights.jsonl` 文件，记录每次权重更新的详细信息：

```json
{"step": 2000, "timestamp": "2025-01-27 10:00:00", "domain_names": ["wiki", "c4"], "domain_weights": [0.3, 0.7], "cumulative_estimated_rewards": [25.3, 45.8], "exploration_rate": 0.0141, "alpha": 0.9, "warmup_steps": 2000, "is_warmup": false}
{"step": 2500, "timestamp": "2025-01-27 10:10:00", "domain_names": ["wiki", "c4"], "domain_weights": [0.25, 0.75], "cumulative_estimated_rewards": [28.1, 52.3], "exploration_rate": 0.0126, "alpha": 0.9, "warmup_steps": 2000, "is_warmup": false}
```

## 权重提取和分析

从训练输出目录中提取优化后的权重：

```python
import json

# 读取权重日志
weights_history = []
with open('odm_result/odm_weights.jsonl', 'r') as f:
    for line in f:
        weights_history.append(json.loads(line))

# 获取最终权重（跳过预热条目）
non_warmup_entries = [e for e in weights_history if not e.get('is_warmup', False)]
if non_warmup_entries:
    final_entry = non_warmup_entries[-1]
    final_weights = final_entry['domain_weights']
    domain_names = final_entry['domain_names']
    
    print("最终优化后的领域权重:")
    for name, weight in zip(domain_names, final_weights):
        print(f"  {name}: {weight:.4f}")

# 可视化权重变化趋势
import matplotlib.pyplot as plt
import numpy as np

non_warmup_steps = [e['step'] for e in non_warmup_entries]
weights_matrix = np.array([e['domain_weights'] for e in non_warmup_entries])

plt.figure(figsize=(10, 6))
for i, name in enumerate(domain_names):
    plt.plot(non_warmup_steps, weights_matrix[:, i], label=name, marker='o')
plt.xlabel('Training Step')
plt.ylabel('Domain Weight')
plt.title('ODM Domain Weight Evolution')
plt.legend()
plt.grid(True)
plt.savefig('odm_weights_evolution.png')
plt.show()

# 分析探索率衰减
exploration_rates = [e['exploration_rate'] for e in non_warmup_entries]
plt.figure(figsize=(10, 4))
plt.plot(non_warmup_steps, exploration_rates, label='Exploration Rate ε_t')
plt.xlabel('Training Step')
plt.ylabel('Exploration Rate')
plt.title('ODM Exploration Rate Decay')
plt.legend()
plt.grid(True)
plt.savefig('odm_exploration_rate.png')
plt.show()
```

## 完整训练示例

```bash
llamafactory-cli train examples/train_full/mixers/odm_dynamic_qwen_pt_full.yaml
```

## 与 DoReMi 的对比

| 方面 | ODM | DoReMi |
|------|-----|--------|
| **参考模型** | 不需要 | 需要（步骤 1） |
| **训练步骤** | 单步 | 三步 |
| **计算成本** | 较低（无参考模型） | 较高（参考 + 代理 + 目标） |
| **适应速度** | 持续在线适应 | 在代理模型上批量优化 |
| **损失信号** | 训练损失 | 过剩损失（相对于参考） |
| **算法** | Exp3（多臂老虎机） | 指数梯度上升 |

## 常见问题

### Q: ODM 与 DoReMi 有何不同？

A: ODM 直接使用训练损失作为奖励信号，在训练过程中在线适应，而 DoReMi 需要参考模型，并在训练目标模型之前在代理模型上优化权重。ODM 更简单易用，但 DoReMi 在使用过剩损失时可能提供更好的理论保证。

### Q: 探索率是如何计算的？

A: 探索率随时间衰减：`ε_t = min{1/K, sqrt(ln(K) / (K * t))}` 其中 K 是领域数量，t 是步数。这确保算法在早期阶段更多探索，在后期阶段更多利用。

### Q: 如果某个领域获得非常低的权重怎么办？

A: 探索率确保每个领域保持至少 `ε_t` 的概率，防止任何领域被完全忽略。随着探索衰减，持续高损失的领域自然会获得较低权重，但它们仍偶尔会被采样。

### Q: 如何在均匀分布和自定义初始比例之间选择？

A: 对于无偏探索，使用均匀分布（`initial_proportions: null`）。如果您对领域重要性有先验知识或想从特定分布开始，请使用自定义比例。算法将从任一起点适应。


## 参考资料

- 论文: [Online Data Mixing: Efficient and Consistent Training for Multilingual Neural Machine Translation](https://arxiv.org/abs/2312.02406)
- 官方实现: [Online Data Mixing GitHub](https://github.com/alon-albalak/online-data-mixing)
- 项目地址: [DataFlex GitHub](https://github.com/OpenDCAI/DataFlex)
