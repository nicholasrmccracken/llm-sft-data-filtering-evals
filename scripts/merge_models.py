import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURATION ---
# 1. The path to the base model you used for training
base_model_path = "Owos/Llama-3.2-1B" 

# 2. The LOCAL path where you downloaded the Tinker adapter
# (Where you ran: tinker checkpoint download ...)
lora_adapter_path = "../models/8104f76f-a448-588c-bd7a-22adc895edad:train:0_sampler_weights_final" 

# 3. Where you want to save the final merged model
output_path = "../models/Llama-3.2-1B-SFT-Merged-Baseline2"

print(f"Loading base model: {base_model_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, # Use bfloat16 to match Llama-3's training
    device_map="cpu",           # Merging is fine on CPU to save VRAM
)

print(f"Loading adapter: {lora_adapter_path}")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

print("Merging weights...")
# This step "bakes" the LoRA matrices into the base weight matrices
merged_model = model.merge_and_unload()

print(f"Saving merged model to: {output_path}")
merged_model.save_pretrained(output_path)

# Also save the tokenizer so olmes can find it
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print("Merge complete!")