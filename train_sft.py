import argparse
import asyncio
import hashlib
import os
from datetime import datetime
from typing import cast

import chz
import datasets
import tinker
import yaml
from dotenv import load_dotenv

from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)


class DatasetFilter:
    """
    Handles optional dataset filtering operations for SFT training:
    - assistant length filtering
    - lexical diversity filtering
    - near-duplicate filtering
    """
    
    def __init__(
        self,
        filter_min_assistant_words: int | None = None,
        filter_max_assistant_words: int | None = None,
        filter_min_lexical_diversity: float | None = None,
        dedup_prefix_words: int | None = None,
    ):
        self.filter_min_assistant_words = filter_min_assistant_words
        self.filter_max_assistant_words = filter_max_assistant_words
        self.filter_min_lexical_diversity = filter_min_lexical_diversity
        self.dedup_prefix_words = dedup_prefix_words

    @staticmethod
    def extract_assistant_text(messages: list[dict]) -> str:
        """
        Concatenate all assistant messages from a conversation.
        """
        parts = []
        for message in messages:
            if message.get("role") == "assistant":
                parts.append(message.get("content", ""))
        return "\n".join(parts).strip()

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text by lowercasing, removing punctuation,
        and collapsing whitespace to single spaces.
        """
        cleaned_chars = []
        for ch in text.lower():
            if ch.isalnum() or ch.isspace():
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append(" ")
        return " ".join("".join(cleaned_chars).split())

    @classmethod
    def assistant_dedup_key(
        cls,
        messages: list[dict],
        prefix_words: int | None = None,
    ) -> str:
        """
        Build a near-duplicate key from the normalized assistant response.
        Optionally truncate to the first N words.
        """
        assistant_text = cls.extract_assistant_text(messages)
        normalized = cls.normalize_text(assistant_text)
        
        if prefix_words is not None:
            normalized = " ".join(normalized.split()[:prefix_words])
            
        return normalized

    def keep_by_length(self, row: dict) -> bool:
        """
        Keep examples whose assistant response length falls within
        the configured word-count bounds.
        """
        if (
            self.filter_min_assistant_words is None
            and self.filter_max_assistant_words is None
        ):
            return True

        assistant_text = self.extract_assistant_text(row["messages"])
        if not assistant_text:
            return False

        word_count = len(assistant_text.split())
        if (
            self.filter_min_assistant_words is not None
            and word_count < self.filter_min_assistant_words
        ):
            return False
        if (
            self.filter_max_assistant_words is not None
            and word_count > self.filter_max_assistant_words
        ):
            return False

        return True

    def keep_by_lexical_diversity(self, row: dict) -> bool:
        """
        Keep examples whose assistant response has lexical diversity
        above the configured minimum threshold.
        
        Lexical diversity is defined as:
            unique_words / total_words
        """
        if self.filter_min_lexical_diversity is None:
            return True

        assistant_text = self.extract_assistant_text(row["messages"])
        if not assistant_text:
            return False

        normalized = self.normalize_text(assistant_text)
        words = normalized.split()

        if not words:
            return False
        
        lexical_diversity = len(set(words)) / len(words)
        return lexical_diversity >= self.filter_min_lexical_diversity

    def apply_dedup(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Apply near-duplicate filtering based on normalized assistant text.
        """
        if self.dedup_prefix_words is None:
            return dataset

        seen_hashes: set[str] = set()

        def keep_unique(row: dict) -> bool:
            key = self.assistant_dedup_key(
                row["messages"],
                prefix_words=self.dedup_prefix_words,
            )
            key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()

            if key_hash in seen_hashes:
                return False

            seen_hashes.add(key_hash)
            return True

        return dataset.filter(keep_unique)
    
    
@chz.chz
class Tulu3Builder(ChatDatasetBuilder):
    """
    Builds the Tulu3 SFT dataset with optional preprocessing.

    Filtering is applied before shuffling and splitting so that
    duplicate examples do not leak across train/test.
    """
    
    filter_min_assistant_words: int | None = None
    filter_max_assistant_words: int | None = None
    filter_min_lexical_diversity: float | None = None
    dedup_prefix_words: int | None = None

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225")
        dataset = cast(datasets.DatasetDict, dataset)["train"]
        original_count = len(dataset)

        dataset_filter = DatasetFilter(
            filter_min_assistant_words=self.filter_min_assistant_words,
            filter_max_assistant_words=self.filter_max_assistant_words,
            filter_min_lexical_diversity=self.filter_min_lexical_diversity,
            dedup_prefix_words=self.dedup_prefix_words,
        )
        
        print("\n--- Dataset Filtering Summary ---")
        print(f"Original examples: {original_count}")
        
        dataset = dataset.filter(dataset_filter.keep_by_length)
        print(f"After length filter: {len(dataset)}")
        
        dataset = dataset.filter(dataset_filter.keep_by_lexical_diversity)
        print(f"After lexical diversity filter: {len(dataset)}")
        
        dataset = dataset_filter.apply_dedup(dataset)
        print(f"After deduplication: {len(dataset)}")
        print(f"Removed total: {original_count - len(dataset)}")
        print("---------------------------------\n")

        if len(dataset) <= 1024:
            raise ValueError(
                "Filtered dataset is too small to create a 1024-example test split."
            )
    
        # Shuffle and split after filtering
        dataset = dataset.shuffle(seed=0)
        test_ds = dataset.take(1024)
        train_ds = dataset.skip(1024)

        # Use train_on_what from common_config if provided, otherwise default to LAST_ASSISTANT_MESSAGE
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        # Take 1024 shuffled examples as test, the rest as train
        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"],
                self.renderer,
                self.common_config.max_length,
                train_on_what,
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds,
            batch_size=self.common_config.batch_size,
            map_fn=map_fn,
        )
        test_dataset = SupervisedDatasetFromHFDataset(
            test_ds,
            batch_size=self.common_config.batch_size,
            map_fn=map_fn,
        )

        print("--- Final Split Sizes ---")
        print(f"Train examples: {len(train_ds)}")
        print(f"Test examples: {len(test_ds)}")
        print("-------------------------\n")

        return train_dataset, test_dataset


class SFTTrainer:
    """
    Wrapper for configuring and running SFT with Tinker.
    """
    def __init__(self, training_args: dict):
        self.training_args = training_args
        self.model_name = training_args.get("model_name", "meta-llama/Llama-3.2-1B")
        self.dataset_name = training_args.get("dataset_name", "tulu3")

        # Renderer determines how conversations are formatted for the model
        self.renderer_name = model_info.get_recommended_renderer_name(self.model_name)
        print(f"Renderer: {self.renderer_name}")

        self.dataset_builder = self._build_dataset_builder()
        self.cookbook_config = self._build_cookbook_config()

    def _build_dataset_builder(self) -> ChatDatasetBuilder:
        """
        Build the dataset loader and preprocessing pipeline from config.
        """
        common_config = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=self.model_name,
            renderer_name=self.renderer_name,
            batch_size=self.training_args.get("batch_size", 32),
            max_length=self.training_args.get("max_length", 2048),
        )

        if self.dataset_name == "tulu3":
            return Tulu3Builder(
                common_config=common_config,
                filter_min_assistant_words=self.training_args.get("filter_min_assistant_words"),
                filter_max_assistant_words=self.training_args.get("filter_max_assistant_words"),
                dedup_prefix_words=self.training_args.get("dedup_prefix_words"),
                filter_min_lexical_diversity=self.training_args.get("filter_min_lexical_diversity"),
            )

        raise ValueError(f"Add a builder for {self.dataset_name}")

    def _build_cookbook_config(self) -> train.Config:
        """
        Construct the Tinker Cookbook training configuration.
        """
        experiment_name = self.training_args.get("experiment_name", "unnamed_run")
        model_short_name = self.model_name.split("/")[-1]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"sft-{model_short_name}-{experiment_name}-{timestamp}"

        return train.Config(
            log_path=f"./logs/{run_name}",
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            dataset_builder=self.dataset_builder,
            evaluator_builders=[],
            infrequent_evaluator_builders=[],
            learning_rate=self.training_args.get("learning_rate", 1e-4),
            lr_schedule="linear",
            num_epochs=self.training_args.get("epochs", 1),
            lora_rank=self.training_args.get("lora_rank", 32),
            save_every=self.training_args.get("save_steps", 100),
            eval_every=self.training_args.get("eval_steps", 100),
            max_steps=self.training_args.get("max_steps"),
            infrequent_eval_every=0,
            rolling_save_every=0,
            wandb_project=None,
            wandb_name=None,
        )

    def train(self):
        """
        Runs the SFT training loop via Tinker Cookbook.
        """
        print(f"Starting optimized training run for {self.model_name}...")
        asyncio.run(train.main(self.cookbook_config))
        print("Training complete! Weights are saved in the log path.")


def main() -> None:
    """
    Entry point for running SFT training from a YAML config file.
    """
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run optimized SFT training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file",
    )
    args = parser.parse_args()

    print(f"Loading experiment settings from: {args.config}")

    # Load YAML config into dictionary
    with open(args.config, "r") as file:
        training_args = yaml.safe_load(file) or {}

    # Use config filename as experiment name
    config_filename = os.path.splitext(os.path.basename(args.config))[0]
    training_args["experiment_name"] = config_filename

    print("\n--- Loaded Configuration ---")
    for key, value in training_args.items():
        print(f"{key}: {value}")
    print("----------------------------\n")

    trainer = SFTTrainer(training_args=training_args)
    trainer.train()


if __name__ == "__main__":
    main()