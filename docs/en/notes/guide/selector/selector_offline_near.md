---
title: Offline-Near-Selector
createTime: 2025/11/27 16:02:41
permalink: /en/guide/7k0w3d92/
icon: flowbite:fish-alt-outline
---
# Offline NEAR Selector 

This document introduces how to use the **Offline NEAR Selector** for **dynamic data selection** during supervised fine-tuning (SFT) within the **DataFlex** framework, finding the most close data to the target dataset to improve generalization performance.

---

## 1. Method Overview

The core idea of **NEAR** is:

* Further encode **already tokenized** samples into **sentence embeddings** (e.g., 512‑dim).
* Perform **nearest‑neighbor search ** in the embedding space to obtain each sample’s representativeness score.

> Intuition: **Closest data for the target dataset** 

### Scoring Formulation

Let the sentence embedding of a sample be $e_i$, and let its $max_K$ nearest neighbors be $\mathcal{N}_K(i)$.



---

## 2. Environment & Dependencies

```bash
# DataFlex (recommended: editable install)
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .

# Common training/inference dependencies (as needed)
pip install llamafactory

# NEAR extras (vector search & progress bars)
pip install faiss-cpu vllm sentence-transformer
```

---

## 3.  Offline Selection

Modify training set, embedding model, and parameters inside
**DataFlex/src/dataflex/offline_selector/offline_near_selector.py**:
```python
if __name__ == "__main__":
    near = offline_near_Selector(
        candidate_path="OpenDCAI/DataFlex-selector-openhermes-10w", # split = train
        query_path="OpenDCAI/DataFlex-selector-openhermes-10w", # split = vaildation
        
        # If you want to use vllm,please add "vllm:" before model's name
        # Otherwise it automatically use sentence-transfromer
        embed_model="vllm:Qwen/Qwen3-Embedding-0.6B",
        batch_size=32,
        save_indices_path="top_indices.npy",
        max_K=1000,
        
    )
    near.selector()
```

Note: model_name is used to encode the already-tokenized text into sentence embeddings (e.g., 512-dim), supporting both vLLM and sentence-transformer inference.

Output: save as the indices matrix that contain the max_K close data for each query
---

## 4. Key Hyperparameters & Tips

| Parameter     | Typical Range | Meaning & Tips                                                                                |
| ------------- | ------------- | --------------------------------------------------------------------------------------------- |
| `max_K`       | 64–10000        | Upper bound of NN retrieval. Larger = stabler but more costly; balance with data size & VRAM. |        |
| `model_name`  | —             | Path/name of the sentence encoder (local BERT/USE/SimCSE, etc.).                              |
| `cache_dir`   | —             | Cache directory for intermediate artifacts and resume‑from‑cache.                             |

---

## 5. Component Config (`components.yaml`)

**Path:** `DataFlex/src/dataflex/configs/components.yaml`

**Preset example**

```yaml
near:
    name: near
    params:
      indices_path: ./src/dataflex/offline_selector/top_indices.npy
      cache_dir: ../dataflex_saves/near_output

```

---

## 6. Dynamic Training Config (LoRA + NEAR)

**Example file:** `DataFlex/examples/train_lora/selectors/near.yaml`

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
component_name: near
warmup_step: 400
update_step: 500
update_times: 2

```

**Notes:**

* `component_name: near` enables the NEAR component.
* `warmup_step / update_step / update_times` decide **when** and **how often** to re‑select the training subset; total steps ≈ `warmup_step + update_step × update_times`.
* total batch_size=device_number x per_device_train_batch_size x gradient_accumulation_steps

---

## 7. Run Training

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/near.yaml
```

**Note:** the above example runs with distributed launch.

During training, NEAR is triggered at scheduled steps: base the sample indice → select the next training subset.

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

We recommend using the [DataFlow](https://github.com/OpenDCAI/DataFlow) QA evaluation pipeline to compare **NEAR** against **Less** and **random sampling**. 


