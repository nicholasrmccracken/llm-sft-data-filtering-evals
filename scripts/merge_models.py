import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "meta-llama/Llama-3.2-1B" 
lora_adapter_path = "/users/PAS3272/chawla114/cse5525-final/trained_models/SFT_Baseline_Rerun"
output_path = "/users/PAS3272/chawla114/cse5525-final/trained_models/Llama-3.2-1B-SFT-Merged-Test-Dataset"

print(f"Loading base model in FP32 for precision...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.float32, # USE FLOAT32 FOR THE ACTUAL MERGE
    device_map="cpu",           
)

print(f"Loading adapter: {lora_adapter_path}")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

print("Merging weights...")
# This 'bakes' the LoRA into the base weights with high precision
merged_model = model.merge_and_unload()

# Optional: Convert back to bfloat16 BEFORE saving to save disk space
merged_model = merged_model.to(torch.bfloat16)

print(f"Saving merged model to: {output_path}")
merged_model.save_pretrained(output_path, safe_serialization=True)

#  Use the Instruct tokenizer to get the chat templates
print("Saving Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B") 
tokenizer.chat_template = "{% for message in messages %}{{ message['role'].capitalize() }}: {{ message['content'] }}\n\n{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"
tokenizer.save_pretrained(output_path)

print("Merge complete! Now try running your evaluations.")