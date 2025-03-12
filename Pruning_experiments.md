# Pruning experiment conducted on Llama family models

### meta-llama/Llama-3.2-1B-Instruct


**changing the layers to prune**

```bash
python llama3.py --pruning_ratio 0.25 \
                 --device cuda --eval_device cuda \
                 --base_model facebook/layerskip-llama3.2-1B \
                 --block_wise --block_mlp_layer_start 2 --block_mlp_layer_end 13 \
                 --block_attention_layer_start 2 --block_attention_layer_end 13 \
                 --save_ckpt_log_name llama3_prune \
                 --pruner_type taylor --taylor param_first \
                 --max_seq_len 2048 \
                 --test_after_train --test_before_train --save_model 

````


### meta-llama/Meta-Llama-3-8B-Instruct

**default parameters**
```bash
python llama3.py --pruning_ratio 0.25 \
                 --device cuda --eval_device cuda \
                 --base_model meta-llama/Meta-Llama-3-8B-Instruct \
                 --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
                 --block_attention_layer_start 4 --block_attention_layer_end 30 \
                 --save_ckpt_log_name llama3_prune \
                 --pruner_type taylor --taylor param_first \
                 --max_seq_len 2048 \
                 --test_after_train --test_before_train --save_model 
```