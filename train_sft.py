import asyncio
from datetime import datetime
from tinker_cookbook import model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook import checkpoint_utils
from dotenv import load_dotenv  # <--- Add this back!
import argparse
import os  
import yaml
class SFTTrainer:
    def __init__(self, training_args):
        self.training_args = training_args
        self.model_name = training_args.get("model_name", "meta-llama/Llama-3.2-1B")
        self.dataset_name = training_args.get("dataset_name", "tulu3")
      
        renderer_name = model_info.get_recommended_renderer_name(self.model_name)
        print("Renderer: "+ renderer_name)
        
        # 1. Setup the Dataset Builder 
        common_config = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=self.model_name,
            renderer_name=renderer_name,  # <--- Added here!
            batch_size=training_args.get("batch_size", 32),
            max_length=training_args.get("max_length", 2048)
        )
        
        if self.dataset_name == "tulu3":
            self.dataset_builder = chat_datasets.Tulu3Builder(common_config=common_config)
        else:
            raise ValueError(f"Add a builder for {self.dataset_name}")

        date_str = datetime.now().strftime("%Y-%m-%d")
        run_name = f"sft-{self.model_name.split('/')[-1]}-{training_args.get("experiment_name", "unnamed_run")}"
        
        # 2. Build the Config object
        self.cookbook_config = train.Config(
            log_path=f"./logs/{run_name}",
            model_name=self.model_name,
            renderer_name=renderer_name, # <--- And added here!
            dataset_builder=self.dataset_builder,
            evaluator_builders=[], # Add eval builders here later if needed
            infrequent_evaluator_builders=[], 
            learning_rate=training_args.get("learning_rate", 1e-4),
            lr_schedule="linear",
            num_epochs=training_args.get("epochs", 1),
            lora_rank=training_args.get("lora_rank", 32),
            save_every=training_args.get("save_steps", 100),
            eval_every=training_args.get("eval_steps", 100),
            max_steps=training_args.get("max_steps", None),
            infrequent_eval_every=0,
            rolling_save_every=0,
            wandb_project=None, 
            wandb_name=None
        )

    def train(self):
        """
        The required train method. It triggers the asynchronous 
        tinker_cookbook loop, bypassing the massive API latency.
        """
        print(f"Starting optimized training run for {self.model_name}...")
        
        # This one line replaces the entire manual forward/backward/step loop
        asyncio.run(train.main(self.cookbook_config))
        
        print("Training complete! Weights are saved in the log path.")

# --- Usage ---
if __name__ == "__main__":
# 1. Catch the config flag from the terminal
    load_dotenv()  # <--- Add this right here!
    parser = argparse.ArgumentParser(description="Run Optimized SFT Training")
    parser.add_argument("--config", type=str, required=True, help="Path to your baseline.yaml")
    args = parser.parse_args()

    # 2. Open and read the YAML configuration file
    print(f"Loading experiment settings from: {args.config}")
    with open(args.config, 'r') as file:
        my_args = yaml.safe_load(file)

    # ---> ADD THESE TWO LINES <---
    # This extracts "baseline" from "configs/baseline.yaml"
    config_filename = os.path.splitext(os.path.basename(args.config))[0]
    my_args["experiment_name"] = config_filename

    # Optional: Print the config to verify it loaded correctly before burning credits
    print("\n--- Loaded Configuration ---")
    for key, value in my_args.items():
        print(f"{key}: {value}")
    print("----------------------------\n")

    # 3. Instantiate your required class with the YAML dictionary
    trainer = SFTTrainer(training_args=my_args)
    
    # 4. Call the required method
    trainer.train()