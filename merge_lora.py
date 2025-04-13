import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# === Paths ===
pruned_model_path = "prune_log/llama3_taylor_param_second/pytorch_model.bin"
lora_path = "tune_log/llama3_taylor_param_second_ft"
output_path = "merged_models/llama3_taylor_param_second_merged"

# === Load config and base model ===
peft_config = PeftConfig.from_pretrained(lora_path)
print(f"Loading base model from: {peft_config.base_model_name_or_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Load LoRA weights ===
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()

# === Save merged model ===
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
tokenizer.save_pretrained(output_path)

print(f"\n✅ Merged model saved to: {output_path}")
