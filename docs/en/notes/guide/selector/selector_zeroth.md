# Introduction to the Zeroth Selector
This document explains how to use the **Zeroth Selector** in the **DataFlex** framework to achieve dynamic selection of training data, thereby enhancing the performance of supervised fine-tuning (SFT). This method is an original selection approach that utilizes co-directional perturbations on the model for differential estimation, thereby obtaining the model's zeroth-order gradient to calculate the effective score of the data.

---

## 1. Method Overview
The core idea of **Zeroth Selector** is:Based on the **SGD** version of the **influence function**, it measures the correlation between training samples and validation samples through the similarity in zeroth-order gradient directions. It is worth noting that for both training and validation data, the perturbation noise uses the same random seed. Therefore, in the current version, the selection algorithm can directly use the product of **differentials (projected gradients)** to represent gradient inner product similarity. Advantages: 1. Avoids the problem of the **gradient-based influence function**, which can only be computed per sample and cannot be parallelized over data; 2. No backpropagation is required, saving time and GPU memory.

### Mathematical Definition
Randomly sample $\xi\sim\mathbf{N}(0,I)$ and perturb the model parameters $\theta$,
$$
\mathrm{Inf}_{\mathrm{zeroth}}(z,z'):=(\dfrac{f(\theta+\epsilon\xi;z)-f(\theta-\epsilon\xi;z)}{2\epsilon})\cdot(\dfrac{f(\theta+\epsilon\xi;z')-f(\theta-\epsilon\xi;z')}{2\epsilon})
$$

## 2. Implementation Steps

### Step 1: Environment Installation

```bash
git clone https://github.com/OpenDCAI/DataFlex.git
cd DataFlex
pip install -e .
pip install llamafactory
```

---
### Step 2: Zeroth Selector Parameter Configuration
**Configuration File Path:**
```
DataFlex/src/dataflex/configs/components.yaml
```
**Example Configuration:**
```yaml
less:
  name: zeroth
  params:
    cache_dir: ../dataflex_saves/zeroth_output
    seed: 42
```

**Parameter Description:** 
* `cache_dir`: The path to save intermediate results, i.e., intermediate difference values.
* `seed`: Optional, random seed used as the generator seed for sampling noise.

---

### Step 3: Dynamic Training Configuration
**Configuration File Path:**
```
DataFlex/examples/train_lora/selectors/zeroth.yaml
```
**Example Configuration:**
```yaml
### model
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 8
#deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: alpaca_en_demo
#dataset: flan_v2,cot_data,dolly_data,oasst1_data
#eval_dataset: mmlu_eval
template: qwen
cutoff_len: 4096
# max_samples: 100000000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 0
# disable_shuffling: true
seed: 42

### output
output_dir: /data1/xlyang/Flex/saves/zeroth/
logging_steps: 10
save_steps: 100
plot_loss: true
save_only_model: false
overwrite_output_dir: true

### swanlab
report_to: none  # choices: [none, wandb, tensorboard, swanlab, 

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: false
fp16: true
ddp_timeout: 180000000

### dynamic_train
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: zeroth
warmup_step: 4
update_step: 3
update_times: 2

eval_dataset: alpaca_zh_demo
```

**Parameter Description:**

* `model_name_or_path`: Model name or path for supervised fine-tuning.
* `dataset`: Training dataset.
* `output_dir`: Output directory of dynamic fine-tuning (LoRA adapter).
* `warmup_step`: Number of warmup steps before the first sample selection.
* `update_step`: Number of steps between each dynamic data selection.
* `update_times`: Total number of dynamic data selection iterations.
* `eval_dataset`: Validation dataset.

Both `dataset` and `eval_dataset` can be selected from `DataFlex/data/dataset_info.json` or local JSON files in ShareGPT/Alpaca format. Note: The training set size significantly affects computation cost. Total steps = `warmup_step + update_step × update_times`.

---

### Step 4: Run Training

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 dataflex-cli train examples/train_lora/selectors/zeroth.yaml
```

The training process automatically performs dynamic data selection and model updates.

---

## 3. Model Evaluation

It is recommended to use the [DataFlow](https://github.com/OpenDCAI/DataFlow) [Model QA Evaluation Pipeline](https://opendcai.github.io/DataFlow-Doc/zh/guide/2k5wjgls/) for systematic evaluation of the generated model.
