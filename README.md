# Multimodal Fine-tuning Lab

Research code for adapting vision-language models with LoRA, including CLIP/LLaVA modules, data processing, evaluation helpers and distributed-training components.

**Status:** experimental components from March 2026. The repository describes fine-tuning existing models; it does not demonstrate training a foundation model from scratch or a production deployment.

## A concrete starting point

The checked-in [training configuration](configs/training_config.yaml) selects `openai/clip-vit-base-patch32`, LoRA rank 16/alpha 32 and COCO-format data. The [CLIP module](src/models/clip_lora.py) loads the base model and processor, applies PEFT adapters and exposes text/image encoding and similarity methods.

```bash
git clone https://github.com/anudeepadi/multimodal-finetuning-lab.git
cd multimodal-finetuning-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use the [data exploration](notebooks/01_data_exploration.ipynb), [training](notebooks/02_model_training.ipynb) and [results visualization](notebooks/03_results_visualization.ipynb) notebooks as the experiment walkthrough. Review their paths and configuration before running. Model weights and datasets are downloaded separately, and GPU/distributed requirements depend on the chosen model and configuration. No training run was executed during this documentation refresh.

## Components

| Directory | Contents |
|---|---|
| `src/models/` | CLIP/LLaVA adaptation and quantization helpers |
| `src/data/` | Dataset loading and preprocessing |
| `src/training/` | Accelerate/distributed trainer components and configuration |
| `src/evaluation/` | Metrics and benchmark helpers |
| `mlops/` | Airflow, MLflow and cloud configuration examples |
| `open-source-contribution/` | Draft contribution proposal and experimental code |

## What needs validation

- No source-backed hardware/run/results record accompanies the earlier efficiency percentages. They have been removed from the project introduction.
- The cloud configuration files are examples, not evidence of deployed services.
- The contribution folder is a proposal, not an accepted upstream contribution. Three Python files there currently contain literal escaped-newline text that fails parsing; they require repair before execution.
- An experiment should save model revision, dataset version/split, seed, hardware, package versions, command and raw metrics. Compare full fine-tuning or frozen embeddings against LoRA under the same evaluation conditions.

The main training modules are components to wire together; there is no verified one-command end-to-end training claim here.
