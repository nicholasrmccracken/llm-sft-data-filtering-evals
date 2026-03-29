"""
This module implements the SFTTrainer class for training your model using supervised fine-tuning (SFT) via Tinker.
"""
import os
import time
import argparse
import yaml
from dotenv import load_dotenv

import tinker
from tinker import types
from datasets import load_dataset

class SFTTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args):
        # 1. THE CONTROL ROOM (Configuration)
        self.model = model  # This holds the Tinker training client
        self.tokenizer = tokenizer
        
        # 2. THE FUEL (Data)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Unpacking the configuration dictionary
        self.training_args = training_args
        self.epochs = training_args.get("epochs", 1)
        self.save_steps = training_args.get("save_steps", 100)
        self.log_steps = training_args.get("log_steps", 10)
        self.lr = training_args.get("learning_rate",0.0002)

    def train(self):
        """
        3. THE ENGINE (Training Loop)
        """
        print("Starting Tinker SFT Engine...")
        global_step = 0

        for epoch in range(self.epochs):
            print(f"\n--- Starting Epoch {epoch + 1}/{self.epochs} ---")
            
            for batch_idx, batch in enumerate(self.train_dataset):
                global_step += 1
                
                # Step A: Forward/Backward Pass (Compute Loss)
                try:
                    loss_future = self.model.forward_backward([batch],loss_fn = "cross_entropy")
                    loss = loss_future.result() 
                    
                    # Step B: Optimizer Step (Update Weights)
                    self.model.optim_step(
                        types.AdamParams(learning_rate=self.lr)
                    ).result()
                    
                except Exception as e:
                    print(f"Network/Compute error at step {global_step}: {e}. Skipping batch...")
                    time.sleep(2)
                    continue 

                # 4. THE DASHBOARD & SAFETY NET (Logging and Checkpointing)
                if global_step % self.log_steps == 0:
                   # Dive into the object path you found!

                    # 1. Get the list
                   # loss_list = loss.loss_fn_outputs
                    #print(loss_list)
                    loss_dict = loss.loss_fn_outputs[0]
                    
                    # 2. Extract the 'elementwise_loss' TensorData object
                    elementwise_tensor = loss_dict["elementwise_loss"]
                    
                    # 3. Get the raw list of 846 numbers from the '.data' attribute
                    loss_array = elementwise_tensor.data
                    
                    # 4. Calculate the average (mean) loss for the sequence
                    if len(loss_array) > 0:
                        avg_loss = sum(loss_array) / len(loss_array)
                    else:
                        avg_loss = 0.0
                        
                    print(f"Epoch: {epoch + 1} | Step: {global_step} | Loss: {avg_loss:.4f}")

        print("\nTraining complete! Your final model weights are saved.")
        self.model.save_state(name="checkpoint_final")


def load_and_format_data(dataset_name, tokenizer, max_samples=None):
    """
    Downloads the Tulu-3 dataset from Hugging Face and converts it to Tinker's format.
    """
    print(f"Downloading {dataset_name} from Hugging Face...")
    hf_dataset = load_dataset(dataset_name, split="train")
    
    if max_samples:
        hf_dataset = hf_dataset.select(range(max_samples))
        print(f"Truncated dataset to {max_samples} samples for fast testing.")

    processed_data = []
    print("Formatting messages into Tinker chunks...")
    
    for row in hf_dataset:
        messages = row["messages"]
        full_text = ""
        
        for msg in messages:
            if msg["role"] == "user":
                full_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                full_text += f"Assistant: {msg['content']}\n"
        
        tokens = tokenizer.encode(full_text.strip())
        
        tinker_input = tinker.ModelInput(
            chunks=[types.EncodedTextChunk(tokens=tokens)]
            
        )

        # 1. The Language Modeling Shift
        # The model sees everything except the last word
        student_input = tokens[:-1] 
        # The model is graded on predicting everything after the first word
        target_tokens = tokens[1:]  
        
        # 2. The Weights Array
        # 1.0 means "calculate the loss for this token". 
        weights = [1.0] * len(target_tokens)
        
        # 3. Your perfectly formatted Datum!
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(student_input),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": weights,
            },
        )
        processed_data.append(datum)
            
    print(f"Successfully formatted {len(processed_data)} examples for Tinker.")
    return processed_data


if __name__ == "__main__":
    # Load environment variables (Make sure your .env file has TINKER_API_KEY=...)
    load_dotenv()

    # 1. Catch the config flag from the bash script
    parser = argparse.ArgumentParser(description="Run SFT Training")
    parser.add_argument("--config", type=str, required=True, help="Path to your config yaml file")
    args = parser.parse_args()

    # 2. Open and read the configuration file
    print(f"Loading experiment settings from: {args.config}")
    with open(args.config, 'r') as file:
        training_args = yaml.safe_load(file)

    # 3. Initialize your Tinker Client
    print("Initializing Tinker connection...")
    service_client = tinker.ServiceClient()
    model = service_client.create_lora_training_client(
        base_model=training_args.get("model_name", "Owos/Llama-3.2-1B"),
        rank=training_args.get("lora_rank", 32)
    )
    tokenizer = model.get_tokenizer()

    # 4. Load your data from Hugging Face
    dataset_name = training_args.get("dataset_name", "allenai/tulu-3-sft-olmo-2-mixture-0225")
    max_samples = training_args.get("max_samples", None) 
    
    train_data = load_and_format_data(
        dataset_name=dataset_name, 
        tokenizer=tokenizer,
        max_samples=max_samples
    )

    # 5. Instantiate your class and press GO
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        val_dataset=[], # Empty for now, but ready for validation data later
        training_args=training_args
    )

    trainer.train()