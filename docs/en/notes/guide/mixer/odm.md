---
title: ODM Data Mixer
createTime: 2025/01/27 10:00:00
icon: material-symbols:casino
permalink: /en/guide/mixer/odm/
---

# ODM Data Mixer

ODM (Online Data Mixing) is an algorithm for dynamically optimizing multi-domain data mixing ratios during training. It uses the Exp3 algorithm from Multi-Armed Bandits to adaptively adjust domain weights based on training loss as a reward signal. Unlike DoReMi, ODM does not require a reference model, making it more efficient and easier to deploy.

## Algorithm Overview

ODM uses the Exp3 (Exponential weights for Exploration and Exploitation) algorithm to dynamically adjust domain weights during training:

1. **Warmup Phase**: Use initial proportions (uniform or specified) for a fixed number of steps
2. **Evaluation Phase**: Evaluate current loss for each domain by sampling batches
3. **Reward Update**: Update cumulative estimated rewards using importance weighting and moving average
4. **Policy Update**: Update sampling policy using Exp3 algorithm with decaying exploration rate

The algorithm automatically balances exploration (trying all domains) and exploitation (focusing on high-reward domains) through a time-decaying exploration rate.

## Key Features

- **No Reference Model Required**: Unlike DoReMi, ODM works directly with the training model
- **Online Learning**: Adapts weights continuously during training
- **Importance Weighting**: Uses importance-weighted rewards to handle non-uniform sampling
- **Decaying Exploration**: Exploration rate decreases over time for better convergence

## Configuration

### Basic Configuration

**Configuration File**: `odm_dynamic_qwen_pt_full.yaml`

```yaml
### dynamic_train - ODM: Online Data Mixing with Multi-Armed Bandits
train_type: dynamic_mix
components_cfg_file: src/dataflex/configs/components.yaml
component_name: odm  # Use ODM mixer
mixture_sample_rule: mixture
init_mixture_proportions: [0.5, 0.5]  # Initial weights
warmup_step: 2000  # Warmup steps before starting ODM
update_step: 500   # Frequency of weight updates
update_times: -1   # -1 means continuous updates until training ends
```

**Configuration in components.yaml**:

```yaml
mixers:
  odm:
    name: odm
    params:
      # Smoothing parameter for exponential moving average (0 to 1)
      alpha: 0.9
      
      # Number of warmup steps using initial proportions
      warmup_steps: 2000
      
      # Number of samples to evaluate per domain when computing rewards
      num_eval_samples: 500
      
      # Batch size for evaluation
      eval_batch_size: 8
      
      # Initial proportions for warmup period
      initial_proportions: [0.5, 0.5]
      # initial_proportions: null  # Use uniform distribution
```

### Training Configuration Parameters

In the training configuration file:

- **`warmup_step`**: Number of steps before starting ODM (should match or exceed `warmup_steps` in components.yaml)
- **`update_step`**: Frequency of weight updates (every N steps)
- **`update_times`**: Number of weight updates. Use `-1` for continuous updates until training ends
- **`train_step`**: Optional explicit total training steps (overrides `num_train_epochs`)

## Training Process

### Single-Step Training

Unlike DoReMi's three-step process, ODM only requires a single training run:

```bash
# Single training run with ODM
llamafactory-cli train examples/train_full/mixers/odm_dynamic_qwen_pt_full.yaml
```

### Weight Update Process

During training, ODM performs the following steps at each update:

1. **Domain Evaluation**: Sample and evaluate batches from each domain to compute current losses
2. **Reward Computation**: Convert losses to rewards (reward = loss / 10.0 to prevent explosion)
3. **Importance Weighting**: Update cumulative estimated rewards using importance weighting: `R̂_i += reward_i / π_i`
4. **Policy Update**: Update domain weights using Exp3 algorithm:
   - Compute exploration rate: `ε_t = min{1/K, sqrt(ln(K) / (K * t))}`
   - Update weights: `w_i = exp(ε_{t-1} * R̂_i) * scaling_factor + ε_t`
   - Normalize to get probabilities: `π_i = w_i / Σ_j w_j`

### Weight Logging

During training, a `odm_weights.jsonl` file is automatically generated, recording detailed information for each weight update:

```json
{"step": 2000, "timestamp": "2025-01-27 10:00:00", "domain_names": ["wiki", "c4"], "domain_weights": [0.3, 0.7], "cumulative_estimated_rewards": [25.3, 45.8], "exploration_rate": 0.0141, "alpha": 0.9, "warmup_steps": 2000, "is_warmup": false}
{"step": 2500, "timestamp": "2025-01-27 10:10:00", "domain_names": ["wiki", "c4"], "domain_weights": [0.25, 0.75], "cumulative_estimated_rewards": [28.1, 52.3], "exploration_rate": 0.0126, "alpha": 0.9, "warmup_steps": 2000, "is_warmup": false}
```

## Weight Extraction and Analysis

Extract optimized weights from the training output directory:

```python
import json

# Read weight logs
weights_history = []
with open('odm_result/odm_weights.jsonl', 'r') as f:
    for line in f:
        weights_history.append(json.loads(line))

# Get final weights (skip warmup entries)
non_warmup_entries = [e for e in weights_history if not e.get('is_warmup', False)]
if non_warmup_entries:
    final_entry = non_warmup_entries[-1]
    final_weights = final_entry['domain_weights']
    domain_names = final_entry['domain_names']
    
    print("Final optimized domain weights:")
    for name, weight in zip(domain_names, final_weights):
        print(f"  {name}: {weight:.4f}")

# Visualize weight evolution
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

# Analyze exploration rate decay
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

## Complete Training Example

```bash
llamafactory-cli train examples/train_full/mixers/odm_dynamic_qwen_pt_full.yaml
```

## Comparison with DoReMi

| Aspect | ODM | DoReMi |
|--------|-----|--------|
| **Reference Model** | Not required | Required (Step 1) |
| **Training Steps** | Single step | Three steps |
| **Computation Cost** | Lower (no reference model) | Higher (reference + proxy + target) |
| **Adaptation Speed** | Continuous online adaptation | Batch optimization on proxy model |
| **Loss Signal** | Training loss | Excess loss (vs reference) |
| **Algorithm** | Exp3 (Multi-Armed Bandits) | Exponentiated Gradient Ascent |

## FAQ

### Q: How does ODM differ from DoReMi?

A: ODM uses training loss directly as a reward signal and adapts online during training, while DoReMi requires a reference model and optimizes weights on a proxy model before training the target model. ODM is simpler to use but DoReMi may provide better theoretical guarantees with excess loss.

### Q: How is the exploration rate computed?

A: The exploration rate decays over time: `ε_t = min{1/K, sqrt(ln(K) / (K * t))}` where K is the number of domains and t is the step number. This ensures the algorithm explores more in early stages and exploits more in later stages.

### Q: What if a domain gets very low weight?

A: The exploration rate ensures each domain maintains at least `ε_t` probability, preventing any domain from being completely ignored. As exploration decays, domains with consistently high loss will naturally get lower weights, but they still get sampled occasionally.

### Q: How to choose between uniform and custom initial proportions?

A: Use uniform distribution (`initial_proportions: null`) for unbiased exploration. Use custom proportions if you have prior knowledge about domain importance or want to start from a specific distribution. The algorithm will adapt from either starting point.

## References

- Paper: [Online Data Mixing: Efficient and Consistent Training for Multilingual Neural Machine Translation](https://arxiv.org/abs/2312.02406)
- Official Implementation: [Online Data Mixing GitHub](https://github.com/alon-albalak/online-data-mixing)
- Project: [DataFlex GitHub](https://github.com/OpenDCAI/DataFlex)
