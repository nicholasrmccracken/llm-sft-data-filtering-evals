from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("/Users/braydenc/cse-5525-spring-2026-default-project-main/cse5525-final/models/Llama-3.2-1B-SFT-Merged-Baseline2", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLama-3.2-1B")

messages = [
    {"role": "user", "content": "What is 2 + 2?"}
]

# Apply the chat template
input_text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True  # adds the assistant turn opener
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=100)

# Decode only the new tokens
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))