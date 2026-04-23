import os
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

base_model_path = os.getenv("BASE_MODEL")
lora_adapter_path = os.getenv("LORA_ADAPTER_PATH")
output_path = os.getenv("OUTPUT_PATH")

if not all([base_model_path, lora_adapter_path, output_path]):
    raise ValueError("Missing BASE_MODEL, LORA_ADAPTER_PATH, or OUTPUT_PATH in .env")

print(f"Loading base model in FP32 for precision...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.float32,
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
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.chat_template = "{% for message in messages %}{{ message['role'].capitalize() }}: {{ message['content'] }}\n\n{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"
tokenizer.save_pretrained(output_path)

print("Merge complete! Now try running your evaluations.")