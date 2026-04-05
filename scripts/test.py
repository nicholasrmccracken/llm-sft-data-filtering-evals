from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("/users/PAS3272/chawla114/cse5525-final/trained_models/Llama-3.2-1B-SFT-Merged-Baseline2-7", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("/users/PAS3272/chawla114/cse5525-final/trained_models/Llama-3.2-1B-SFT-Merged-Baseline2-7")

messages = [
    {"role": "user", "content": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"}
]

# Apply the chat template
input_text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True  # adds the assistant turn opener
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=2048)

# Decode only the new tokens
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))