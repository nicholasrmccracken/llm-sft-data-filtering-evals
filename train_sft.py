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
from transformers import AutoTokenizer

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


@chz.chz
class Tulu3Builder(ChatDatasetBuilder):
    """
    Builds the Tulu3 SFT dataset with optional preprocessing.

    Supported preprocessing:
    - assistant response length filtering
    - exact deduplication on normalized conversation text

    Filtering is applied before shuffling and splitting so that
    duplicate examples do not leak across train/test.
    """
    filter_min_assistant_tokens: int | None = None
    filter_max_assistant_tokens: int | None = None
    dedup_exact: bool = False

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
        Normalize text for exact deduplication by lowercasing and
        collapsing all whitespace to single spaces.
        """
        return " ".join(text.lower().split())

    @classmethod
    def conversation_key(cls, messages: list[dict]) -> str:
        """
        Create a normalized string representation of a full conversation.
        """
        parts = []
        for message in messages:
            role = message.get("role", "").strip().lower()
            content = cls.normalize_text(message.get("content", ""))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225")
        dataset = cast(datasets.DatasetDict, dataset)["train"]
        tokenizer = AutoTokenizer.from_pretrained(
            self.common_config.model_name_for_tokenizer
        )
        original_count = len(dataset)

        def keep_example(row: dict) -> bool:
            """
            Keep examples whose assistant response length falls within
            the configured token bounds. Examples with no assistant
            response are dropped.
            """
            assistant_text = self.extract_assistant_text(row["messages"])

            if not assistant_text: return False
            if (
                self.filter_min_assistant_tokens is None
                and self.filter_max_assistant_tokens is None
            ): return True

            token_count = len(
                tokenizer(
                    assistant_text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )

            if (
                self.filter_min_assistant_tokens is not None
                and token_count < self.filter_min_assistant_tokens
            ): return False
            if (
                self.filter_max_assistant_tokens is not None
                and token_count > self.filter_max_assistant_tokens
            ): return False

            return True

        dataset = dataset.filter(keep_example)
        after_length_filter_count = len(dataset)

        if self.dedup_exact:
            seen_hashes = set()

            def keep_unique(row: dict) -> bool:
                """
                Keep only the first occurrence of each normalized conversation.
                """
                key = self.conversation_key(row["messages"])
                key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
                if key_hash in seen_hashes: return False

                seen_hashes.add(key_hash)
                return True

            dataset = dataset.filter(keep_unique)

        final_count = len(dataset)

        print("\n--- Dataset Filtering Summary ---")
        print(f"Original examples: {original_count}")
        print(f"After length filter: {after_length_filter_count}")
        print(f"After deduplication: {final_count}")
        print(f"Removed total: {original_count - final_count}")
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
                filter_min_assistant_tokens=self.training_args.get("filter_min_assistant_tokens"),
                filter_max_assistant_tokens=self.training_args.get("filter_max_assistant_tokens"),
                dedup_exact=self.training_args.get("dedup_exact", False),
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