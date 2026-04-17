---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: qwen2.5-1.5b-gsm8k-lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen2.5-1.5b-gsm8k-lora

This model is a fine-tuned version of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on the train_data_gsm8k dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1309

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- total_eval_batch_size: 8
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.0699        | 2.3855 | 100  | 0.1292          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.52.4
- Pytorch 2.7.1+cu118
- Datasets 3.6.0
- Tokenizers 0.21.1