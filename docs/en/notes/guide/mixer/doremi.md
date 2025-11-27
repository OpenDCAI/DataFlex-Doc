---
title: DoReMi Data Mixer
createTime: 2025/11/27 10:00:00
icon: material-symbols:balance
permalink: /en/guide/mixer/doremi/
---

# DoReMi Data Mixer

DoReMi (Domain Reweighting with Minimax Optimization) is an algorithm for optimizing multi-domain data mixing ratios. By performing domain weight optimization on a small proxy model, it can find the optimal data mixing strategy for large-scale model training.

## Algorithm Overview

The DoReMi algorithm consists of three steps:

1. **Step 1**: Train a reference model using reference weights
2. **Step 2**: Dynamically optimize domain weights on a small proxy model
3. **Step 3**: Train the large-scale target model using optimized weights

## Three-Step Training Process

### Step 1: Reference Model Training

Train a reference model using initial domain weights (typically uniform distribution or empirical weights). This model serves as the baseline for subsequent weight optimization.

**Configuration File**: `doremi_step1_static_qwen_pt_full.yaml`

```yaml
### dynamic_train - DoReMi Step 1: Reference Model Training
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: static  # Use static mixer
mixture_sample_rule: mixture
init_mixture_proportions: [0.5, 0.5]  # Initial weights, uniform distribution
static_mix: true
```

**Key Parameters**:
- `component_name: static`: Use static mixer to keep weights unchanged throughout training
- `init_mixture_proportions`: Initial domain weights, must match the number of datasets
- `static_mix: true`: Enable static mixing mode

**Configuration in components.yaml**:

```yaml
mixers:
  static:
    name: static
    params:
      proportions: [0.5, 0.5]  # Proportion for each domain
      # proportions: null  # Use uniform distribution
```

### Step 2: Proxy Model Weight Optimization

Use the DoReMi algorithm to dynamically optimize domain weights on a small proxy model. The algorithm adjusts weights by computing excess loss for each domain. During training, the algorithm uses uniform sampling for data selection, but the optimized domain weights are recorded and used for loss reweighting in the training step.

**Configuration File**: `doremi_step2_dynamic_qwen_pt_full.yaml`

```yaml
### dynamic_train - DoReMi Step 2: Proxy Model Training
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: doremi  # Use DoReMi mixer
mixture_sample_rule: mixture
init_mixture_proportions: [0.5, 0.5]  # Initial weights
warmup_step: 100  # Warmup steps
update_step: 200  # Weight update interval
update_times: 3   # Number of weight updates
```

**Configuration in components.yaml**:

```yaml
mixers:
  doremi:
    name: doremi
    params:
      # Reference model path from Step 1
      reference_model_path: /path/to/doremi_step1_result/checkpoint-xxx
      # Weight update learning rate (eta in DoReMi paper)
      reweight_eta: 0.1
      # Weight smoothing parameter (epsilon in DoReMi paper)
      reweight_eps: 0.01
```

**Key Parameters**:
- `reference_model_path`: Path to the reference model checkpoint from Step 1
- `reweight_eta`: Learning rate for weight updates, controls adjustment magnitude
- `reweight_eps`: Smoothing parameter to prevent domain weights from becoming too small
- `warmup_step`: Number of warmup training steps before starting weight optimization
- `update_step`: Frequency of weight updates (every N steps)

**Algorithm Behavior**:
- The algorithm uses **uniform sampling** for data selection (each domain has equal probability)
- The optimized `domain_weights` are computed and used for **loss reweighting** during training
- This approach ensures fair sampling while allowing the loss function to focus on harder domains

**Weight Logging**:

During training, a `doremi_weights.jsonl` file is automatically generated, recording detailed information for each weight update:

```json
{"step": 100, "timestamp": "2025-11-27 10:00:00", "domain_names": ["wiki", "c4"], "domain_weights": [0.3, 0.7], "perdomain_scores": [2.5, 3.2]}
{"step": 300, "timestamp": "2025-11-27 10:10:00", "domain_names": ["wiki", "c4"], "domain_weights": [0.25, 0.75], "perdomain_scores": [2.3, 3.5]}
```

### Step 3: Target Model Training

Train the large-scale target model using the final optimized weights from Step 2.

**Configuration File**: `doremi_step3_static_qwen_pt_full.yaml`

```yaml
### dynamic_train - DoReMi Step 3: Large Model Training with Optimized Weights
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: static  # Use static mixer
mixture_sample_rule: mixture
init_mixture_proportions: [0.3, 0.7]  # Use optimized weights from Step 2
static_mix: true
```

**Key Steps**:
1. Extract the final optimized weights from Step 2's `doremi_weights.jsonl` file
2. Fill the weights into the `init_mixture_proportions` configuration
3. Train using the static mixer

## Complete Training Example

```bash
# Step 1: Train reference model
llamafactory-cli train examples/train_full/mixers/doremi_step1_static_qwen_pt_full.yaml

# Step 2: Optimize domain weights (on small proxy model)
# Note: Update reference_model_path in components.yaml first
llamafactory-cli train examples/train_full/mixers/doremi_step2_dynamic_qwen_pt_full.yaml

# Step 3: Train target model with optimized weights
# Note: Fill in final weights from Step 2 into config file
llamafactory-cli train examples/train_full/mixers/doremi_step3_static_qwen_pt_full.yaml
```

## Weight Extraction and Analysis

Extract optimized weights from Step 2's output directory:

```python
import json

# Read weight logs
weights_history = []
with open('doremi_step2_result/doremi_weights.jsonl', 'r') as f:
    for line in f:
        weights_history.append(json.loads(line))

# Get final weights
final_weights = weights_history[-1]['domain_weights']
domain_names = weights_history[-1]['domain_names']

print("Optimized domain weights:")
for name, weight in zip(domain_names, final_weights):
    print(f"  {name}: {weight:.4f}")

# Visualize weight evolution
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

## Best Practices

### 1. Reference Model Training

- Use uniform distribution or dataset-size-based proportions as initial weights
- Ensure the reference model converges sufficiently, recommend at least one full epoch
- Save multiple checkpoints and select the model with lowest validation loss

### 2. Weight Optimization

- Recommend using small proxy models (e.g., 0.5B-1B parameters) to reduce computational cost
- `reweight_eta` can be adjusted based on convergence (higher values lead to faster weight changes)
- `reweight_eps` controls the minimum weight for each domain
- Recommend observing convergence trends to set appropriate number of weight updates (`update_times`)
- The algorithm uses uniform sampling but applies domain weights to loss reweighting

### 3. Target Model Training

- Use weights from the last update in Step 2, not intermediate results
- Compare performance between optimized weights and uniform distribution
- Evaluate model performance on downstream tasks

## FAQ

### Q: Why are three steps needed?

A: DoReMi's core idea is to optimize weights by comparing losses between reference and proxy models. Step 1 provides the baseline, Step 2 quickly finds optimal weights on a small model, and Step 3 applies results to large model training.

### Q: How are weights updated?

A: Using Exponentiated Gradient Ascent algorithm. Domains with higher excess loss get increased weights; those with lower excess loss get decreased weights. Formula:

$$
w_i^{(t+1)} \propto w_i^{(t)} \cdot \exp(\eta \cdot \text{excess\_loss}_i^{(t)})
$$

### Q: How to choose initial weights?

A: Options include:
- Uniform distribution: `[1/k, 1/k, ..., 1/k]`
- Proportions based on dataset sizes
- Proportions based on domain prior knowledge

### Q: Can it run without a reference model?

A: Yes. If `reference_model_path` is set to `null`, the algorithm will directly use proxy model losses for optimization (equivalent to minimizing training loss). However, note that this is not part of the DoReMi algorithm, so it's only recommended for debugging purposes.

## References

- Paper: [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429)
- Project: [DataFlex GitHub](https://github.com/OpenDCAI/DataFlex)

