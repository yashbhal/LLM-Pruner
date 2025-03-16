# 🔪 LLM Pruning Experiments: Llama Family Models

> This document contains configuration and commands for pruning experiments conducted on various Llama family models.

## Table of Contents
- [Meta-Llama-3.2-1B-Instruct](#meta-llama32-1b-instruct)
- [Meta-Llama-3-8B-Instruct](#meta-llama-3-8b-instruct)
- [Post-Pruning Fine-Tuning](#post-pruning-fine-tuning)

---
## Meta-Llama-3.2-1B-Instruct - 0.35 Pruning Ratio

### 🔄 Changing the Layers to Prune

```bash
python llama3.py \
    --pruning_ratio 0.35 \
    --device cuda \
    --eval_device cuda \
    --base_model facebook/layerskip-llama3.2-1B \
    --block_wise \
    --block_mlp_layer_start 2 \
    --block_mlp_layer_end 13 \
    --block_attention_layer_start 2 \
    --block_attention_layer_end 13 \
    --save_ckpt_log_name llama3_prune \
    --pruner_type taylor \
    --taylor param_first \
    --max_seq_len 2048 \
    --test_after_train \
    --test_before_train \
    --save_model
```

---
## Post-Pruning Fine-Tuning

> **Note:** Post-pruning fine-tuning is required to restore accuracy.

```bash
CUDA_VISIBLE_DEVICES=0 python post_training.py \
    --prune_model prune_log/llama3_prune/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --output_dir tune_log/llama3-tune \
    --wandb_project llama_tune
```





## layerskip-llama3.2-1B - Layerskip 

### 🔄 Changing the Layers to Prune

```bash
python llama3.py \
    --pruning_ratio 0.25 \
    --device cuda \
    --eval_device cuda \
    --base_model facebook/layerskip-llama3.2-1B \
    --block_wise \
    --block_mlp_layer_start 2 \
    --block_mlp_layer_end 13 \
    --block_attention_layer_start 2 \
    --block_attention_layer_end 13 \
    --save_ckpt_log_name layerskip_1b_prune_0.25 \
    --pruner_type taylor \
    --taylor param_mix \
    --max_seq_len 2048 \
    --test_after_train \
    --test_before_train \
    --save_model
```

---

## Meta-Llama-3.2-1B-Instruct - Original Llama3.2 1b

### 🔄 Changing the Layers to Prune

```bash
python llama3.py \
    --pruning_ratio 0.25 \
    --device cuda \
    --eval_device cuda \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --block_wise \
    --block_mlp_layer_start 2 \
    --block_mlp_layer_end 13 \
    --block_attention_layer_start 2 \
    --block_attention_layer_end 13 \
    --save_ckpt_log_name vanilla_llama_1b_prune_0.25 \
    --pruner_type taylor \
    --taylor param_mix \
    --max_seq_len 2048 \
    --test_after_train \
    --test_before_train \
    --save_model
```

---

## Meta-Llama-3-8B-Instruct

### ⚙️ Default Parameters

```bash
python llama3.py \
    --pruning_ratio 0.25 \
    --device cuda \
    --eval_device cuda \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --block_wise \
    --block_mlp_layer_start 4 \
    --block_mlp_layer_end 30 \
    --block_attention_layer_start 4 \
    --block_attention_layer_end 30 \
    --save_ckpt_log_name llama3_prune \
    --pruner_type taylor \
    --taylor param_first \
    --max_seq_len 2048 \
    --test_after_train \
    --test_before_train \
    --save_model
```

---

## Post-Pruning Fine-Tuning

> **Note:** Post-pruning fine-tuning is required to restore accuracy.

```bash
CUDA_VISIBLE_DEVICES=0 python post_training.py \
    --prune_model prune_log/vanilla_llama_1b_prune_0.25/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --output_dir tune_log/llama3-tune \
    --wandb_project vanilla_llama_0.25_finetune
```





# Chat with models on a web inferface

### original model
```bash
 python generate.py --model_type pretrain
```
### pruned model
```bash
python generate.py \
--model_type pruneLLM \
--ckpt prune_log/vanilla_llama_1b_prune_0.25/pytorch_model.bin 
```
### prune + fine-tune
```bash
python generate.py \
--model_type tune_prune_LLM \
--ckpt prune_log/vanilla_llama_1b_prune_0.25/pytorch_model.bin \
--lora_ckpt tune_log/llama3-tune
```


# Evaluations

## Evaluating on the pruned+fine_tuned model
```bash
export PYTHONPATH='.'
python lm-evaluation-harness/main.py --model hf-causal-experimental \
       --model_args checkpoint=prune_log/vanilla_llama_1b_prune_0.25/pytorch_model.bin,peft=tune_log/llama3-tune,config_pretrained=meta-llama/Llama-3.2-1B-Instruct \
       --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
       --device cuda:0 --no_cache \
       --output_path my_evaluations/llama3.2_1b_0.25_eval
```

