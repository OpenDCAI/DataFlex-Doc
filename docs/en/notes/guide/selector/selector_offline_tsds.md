---
title: Offline-Tsds-Selector
createTime: 2025/11/01 21:36:21
permalink: /en/guide/im5q9cd2/
icon: tdesign:cat
---


# Offline TSDS Selector 

This document introduces how to use the **Offline TSDS Selector** for **dynamic data selection** during supervised fine-tuning (SFT) within the **DataFlex** framework, achieving a balance between **density representativeness** and **diversity** to improve generalization performance.

---

## 1. Method Overview

The core idea of **TSDS** is:

* Further encode **already tokenized** samples into **sentence embeddings** (e.g., 512‑dim).
* Perform **nearest‑neighbor search & kernel density estimation (KDE)** in the embedding space to obtain each sample’s representativeness score.
* Incorporate **topological diversity** (avoid only picking clusters), and trade off density vs. diversity via the coefficient `alpha`.

> Intuition: **Higher density** ⇒ more “typical/representative” samples; **higher diversity** ⇒ broader coverage and less redundancy.

### Scoring Formulation

Let the sentence embedding of a sample be $e_i$, and let its $K$ nearest neighbors be $\mathcal{N}_K(i)$.


1. **Kernel Density Estimation (KDE):**
   $$
   \text{density}(i)
   = \frac{1}{K} \sum_{j\in \mathcal{N}_K(i)}
   \exp!\left(-\frac{\lVert e_i - e_j \rVert^2}{2\sigma^2}\right)
   $$

2. **Diversity (simple implementation via de‑dup penalty / marginal gain):**
   $$
   \text{diversity}(i)\ \propto
   \min_{j\in S} \lVert e_i - e_j \rVert,\quad
   S=\text{selected set}
   $$

3. **Combined Score:**
   $$
   \text{score}(i)
   = \alpha, \text{density}(i)

   * (1-\alpha), \text{diversity}(i)
   $$

> In practice, `kde_K` (neighbors used by KDE) and `max_K` (overall NN search limit) can differ. `C` can be used as a selection ratio/threshold or other control term depending on the implementation.

---

## 2. Environment & Dependencies

```bash
# DataFlex (recommended: editable install)
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .

# Common training/inference dependencies (as needed)
pip install llamafactory

# TSDS extras (vector search & progress bars)
pip install faiss-cpu vllm sentence-transformer
```

---

## 3.  Offline Selection

Modify training set, embedding model, and parameters inside
**DataFlex/src/dataflex/offline_selector/offline_tsds_selector.py**:
```python
if __name__ == "__main__":
    tsds = offline_tsds_Selector(
        candidate_path="OpenDCAI/DataFlex-selector-openhermes-10w", # training set
        query_path="OpenDCAI/DataFlex-selector-openhermes-10w",     # validation set

        # If you want to use vllm, please add "vllm:" before the model name
        # Otherwise it automatically uses sentence-transformer
        embed_model="vllm:Qwen/Qwen3-Embedding-0.6B", # embedding model
        batch_size=32,
        save_probs_path="tsds_probs.npy",
        max_K=5000,
        kde_K=1000,
        sigma=0.75,
        alpha=0.6,
        C=5.0
    )
    tsds.selector()
```

Note: model_name is used to encode the already-tokenized text into sentence embeddings (e.g., 512-dim), supporting both vLLM and sentence-transformer inference.

Output: a sampling probability for each training sample.
---

## 4. Key Hyperparameters & Tips

| Parameter     | Typical Range | Meaning & Tips                                                                                |
| ------------- | ------------- | --------------------------------------------------------------------------------------------- |
| `max_K`       | 64–10000        | Upper bound of NN retrieval. Larger = stabler but more costly; balance with data size & VRAM. |
| `kde_K`       | 16–2000         | #neighbors in KDE. Smaller = more sensitive; larger = smoother. Usually `kde_K ≤ max_K`.      |
| `sigma`       | 0.5–2.0       | KDE bandwidth. Too small ⇒ noisy; too large ⇒ oversmoothing.                                  |
| `alpha`       | 0.3–0.7       | Trade‑off between representativeness (density) and coverage (diversity).                      |
| `C`           | 0.01–1.0      | Selection ratio/threshold or regularization strength depending on implementation.             |
| `sample_size` | 500–5000      | Candidate pool size per selection step; heavily impacts speed & quality.                      |
| `model_name`  | —             | Path/name of the sentence encoder (local BERT/USE/SimCSE, etc.).                              |
| `cache_dir`   | —             | Cache directory for intermediate artifacts and resume‑from‑cache.                             |

---

## 5. Component Config (`components.yaml`)

**Path:** `DataFlex/src/dataflex/configs/components.yaml`

**Preset example**

```yaml
tsds:
    name: tsds
    params:
      probs_path: ./src/dataflex/offline_selector/tsds_probs.npy 
      # default location of tsds_probs.npy
      cache_dir: ../dataflex_saves/tsds_output

```

---

## 6. Dynamic Training Config (LoRA + TSDS)

**Example file:** `DataFlex/examples/train_lora/selectors/tsds.yaml`

```yaml
### model
model_name_or_path: 
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 8

### dataset
dataset: # training dataset
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ../dataflex_saves
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### Dataflex args
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: tsds
warmup_step: 400
update_step: 500
update_times: 2

```

**Notes:**

* `component_name: tsds` enables the TSDS component.
* `warmup_step / update_step / update_times` decide **when** and **how often** to re‑select the training subset; total steps ≈ `warmup_step + update_step × update_times`.
* total batch_size=device_number x per_device_train_batch_size x gradient_accumulation_steps

---

## 7. Run Training

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/tsds.yaml
```

**Note:** the above example runs with distributed launch.

During training, TSDS is triggered at scheduled steps: base the sample probablity → select the next training subset.

---

## 8. Merge & Export the Model

Same as the Less Selector pipeline.

**Config file:** `DataFlex/examples/merge_lora/llama3_lora_sft.yaml`

```yaml
model_name_or_path: base model path
adapter_name_or_path: finetuned adapter path
template: qwen
trust_remote_code: true

export_dir: ../dataflex_saves
export_size: 5
export_device: cpu
export_legacy_format: false

```

Run the export command (inside the LLaMA‑Factory directory):

```bash
llamafactory-cli export llama3_lora_sft.yaml
```

---

## 9. Evaluation & Comparison

We recommend using the [DataFlow](https://github.com/OpenDCAI/DataFlow) QA evaluation pipeline to compare **TSDS** against **Less** and **random sampling**. 


