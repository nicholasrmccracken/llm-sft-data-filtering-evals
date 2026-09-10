# LLM Fine-Tuning & Data Filtering

## Overview

This project investigates how instruction-data filtering affects supervised fine-tuning performance for small language models.

The pipeline fine-tunes **Llama 3.2 1B** using configurable filtering strategies applied to instruction-following data before training. The main filters include response-length constraints, lexical-diversity thresholds, and near-duplicate removal. Each filtered dataset can then be used for supervised fine-tuning with LoRA, followed by model merging and benchmark evaluation.

The goal is to compare how different data-selection strategies influence model quality, training behavior, and downstream benchmark performance.

## Project Structure

```text
├── train_sft.py              # Supervised fine-tuning pipeline
├── configs/                  # Training and filtering configurations
├── scripts/                  # Training, merging, and evaluation utilities
├── evals/                    # OLMES evaluation suite
├── eval_outputs/             # Benchmark outputs
├── logs/                     # Training logs and saved runs
├── Proposal_5525.pdf         # Initial project proposal
├── Midpoint_5525.pdf         # Midpoint project report
└── README.md
```

## Data Filtering

The training pipeline supports several configurable preprocessing strategies.

### Response-Length Filtering

Examples can be filtered based on the length of the assistant response.

This allows training runs to exclude responses that are either too short to contain meaningful instruction-following behavior or unnecessarily long for the target training setup.

Supported configuration options include:

```yaml
filter_min_assistant_words:
filter_max_assistant_words:
```

### Lexical-Diversity Filtering

The pipeline can remove examples with low lexical diversity.

Lexical diversity is calculated as:

```text
unique words / total words
```

This provides a simple way to filter repetitive responses before fine-tuning.

```yaml
filter_min_lexical_diversity:
```

### Near-Duplicate Filtering

Near-duplicate assistant responses can also be removed before the train/test split.

Responses are normalized by:

* converting text to lowercase
* removing punctuation
* collapsing whitespace
* optionally comparing only the first N words
* hashing the normalized text to detect duplicates

```yaml
dedup_prefix_words:
```

Filtering is performed before shuffling and splitting so that duplicate responses do not leak between the training and evaluation datasets.

## Supervised Fine-Tuning

The project fine-tunes **Meta Llama 3.2 1B** using supervised fine-tuning through the Tinker training stack.

Training is configurable through YAML files and supports parameters such as:

```yaml
model_name:
batch_size:
max_length:
learning_rate:
epochs:
lora_rank:
save_steps:
eval_steps:
max_steps:
```

The pipeline automatically:

1. Loads the instruction dataset
2. Applies the configured filtering strategy
3. Shuffles the filtered dataset
4. Creates training and evaluation splits
5. Formats conversations for the selected model
6. Fine-tunes the model using LoRA
7. Saves checkpoints and training logs

## Dataset

The project uses the **Tulu 3 SFT mixture**:

```text
allenai/tulu-3-sft-olmo-2-mixture-0225
```

A fixed set of 1,024 shuffled examples is reserved for evaluation after filtering, while the remaining examples are used for training.

## Training Experiments

The project is structured around comparing multiple data-selection strategies against an unfiltered baseline.

Example experiment categories include:

* Baseline SFT
* Response-length filtering
* Lexical-diversity filtering
* Near-duplicate filtering
* Combined filtering strategies

Each experiment can be configured independently, allowing training behavior and evaluation results to be compared across runs.

## LoRA Adapter Merging

Fine-tuning is performed using **LoRA** to reduce the number of trainable parameters.

After training, LoRA adapters can be merged back into the base model weights to create a standalone model checkpoint for evaluation.

The resulting merged model can then be loaded by standard Hugging Face tooling and evaluated through OLMES.

## Evaluation

Fine-tuned models are evaluated using **OLMES**, AI2's Open Language Model Evaluation System.

Benchmarks include:

| Benchmark     | Focus                       |
| ------------- | --------------------------- |
| **GSM8K**     | Mathematical reasoning      |
| **IFEval**    | Instruction following       |
| **MBPP**      | Python code generation      |
| **HarmBench** | Safety evaluation           |
| **XSTest**    | Safety and refusal behavior |

Evaluation outputs include model predictions and aggregated benchmark metrics.

```text
eval_outputs/
├── gsm8k-predictions.jsonl
├── ifeval-predictions.jsonl
├── mbpp-predictions.jsonl
└── harmbench-predictions.jsonl
```

## Running Training

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
uv pip install tinker
```

Configure a training run using a YAML file in `configs/`.

Then run:

```bash
python train_sft.py --config configs/<config-name>.yaml
```

The configuration filename is automatically incorporated into the experiment name and training log directory.

## Running Evaluation

Set up OLMES:

```bash
cd evals/olmes

uv sync
uv sync --group gpu
```

Then run an evaluation from the `evals` directory:

```bash
olmes --model <model-path> --task gsm8k --output-dir <output-dir>
```

For example:

```bash
olmes --model <model-path> --task ifeval --output-dir <output-dir>
```

Multiple evaluation tasks can be run to compare the performance of models trained using different filtering strategies.

## Tech Stack

* **Python**
* **PyTorch**
* **Hugging Face**
* **Tinker**
* **LoRA / PEFT**
* **OLMES**
* **Datasets**
* **YAML**

## Key Ideas

This project focuses on a simple question:

**Can better instruction-data selection improve fine-tuning outcomes without changing the underlying model architecture?**

Instead of treating the entire training dataset as equally valuable, the pipeline makes data quality a configurable part of the fine-tuning process and provides a reproducible way to compare different filtering approaches.
