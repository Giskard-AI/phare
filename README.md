# Phare Benchmark

Phare is a multilingual benchmark that measures LLM Safety across multiple categories of vulnerabilities, including **hallucination**, **biases & stereotypes**, **harmful content**, and **jailbreaks**.

![Phare Benchmark](images/phare.png)

## Background and Motivation

Large Language Models (LLMs) have rapidly advanced in capability and adoption, becoming essential tools across a wide spectrum of natural language processing applications. While existing benchmarks have focused primarily on general performance metrics, such as accuracy and task completion, there is a growing recognition of the need to evaluate these models through the lens of safety and robustness. Concerns over hallucination, harmful outputs, and social bias have escalated alongside model deployment in sensitive and high-impact settings.

However, current safety evaluations tend to be fragmented or limited in scope, lacking unified diagnostic tools that deeply probe model behavior. In response to this gap, we introduce Phare, a multilingual and multi-faceted evaluation framework designed specifically to diagnose and analyze LLM failure modes across hallucination, bias, and harmful content. Phare aims at contributing a tool for the development of safer and more trustworthy AI systems.

## Usage

Phare is easy to use and reproducible. You can set up and run the benchmark with just a few commands using the `uv` package manager.

## Setup
### Clone this repository
```
git clone https://github.com/Giskard-AI/phare.git
cd phare
uv sync
```

### Fetch the data from Hugging Face
```
uv run python download_phare.py --path ./phare_data
```

### Set up API Keys
Set up the API keys for the providers you want to use in the `.env` file.
Please note that the default configuration uses three judges models to evaluate the samples: 

- GPT 5 mini
- Gemini 2.5 Flash Lite
- Claude 4.5 Haiku

To run the benchmark with this configuration, you need at least the following API keys to be set in the `.env` file:
- OPENAI_API_KEY
- GEMINI_API_KEY
- ANTHROPIC_API_KEY

## Run the benchmark
```
uv run --env-file .env flare --config-path example_config.json --sample-path ./phare_data/public_set --name phare_public_set
```

Optionnally, you can use the `--max-samples-per-task` argument to limit the number of samples to run for each task.

### Run output structure

The results of a run will be saved in the `runs/<experiment_name>` folder. The structure of the results is the following:

- `generate/`: Contains the generation results from the models under test.
- `error/`: Contains all the samples that caused an error during the generation or evaluation process.
- `result/`: Contains all the samples that were correctly evaluated.

Each of these folders is organized by model and modules, and inside each model folder, you will find one file per sample named `<sample_id>.json`.

### Custom configuration

You can customize the configuration by creating a JSON file with two sections:

#### Models
Define the LLMs to evaluate:
```json
{
    "models": [
        {
            "name": "GPT 5 mini",
            "litellm_model": "openai/gpt-5-mini",
            "reasoning_effort": "low"
        }
    ]
}
```
- `name`: Display name
- `litellm_model`: Model identifier (LiteLLM format: `provider/model-name`)
- Additional generation arguments (`temperature`, `thinking`, `reasoning_effort`, etc.) can be passed directly.

#### Scorers
Configure the LLM-as-a-judge evaluators. Available scorers:
- **Hallucination:** `factuality`, `misinformation`, `debunking`, `tools`
- **Harmful:** `harmful_misguidance`
- **Jailbreak:** `jailbreak/encoding`, `jailbreak/framing`, `jailbreak/injection`
- **Biases:** `biases/story_generation`

**Scorer names map directly to sample categories in the Phare dataset.** Adding or removing a scorer from your config will include or exclude the corresponding samples from evaluation.

Each scorer uses an ensemble of judge models:
```json
{
    "scorers": {
        "factuality": {
            "parallelism": 15,
            "models": [
                {"litellm_model": "openai/gpt-5-mini", "weight": 1.0}
            ]
        }
    }
}
```

See `example_config.json` for a complete reference.

## Data Structure

Each sample in the Phare dataset is stored as a JSON object in JSONL files:

```jsonc
{
    "id": "c3a6b204-fc01-4a22-2ed0-b0b27da56b6e",  // Unique sample identifier
    "module": "hallucination",  // Category: hallucination, biases, harmful, jailbreak
    "task": "debunking",        // Specific task within the module
    "language": "es",           // Language code (en, fr, es, ...)
    "generations": [            // List of generation requests (single element for most tasks)
        {
            "id": "1e7219b5-fb3d-45ff-ae71-ba2bd835d33d",
            "type": "chat_completion",
            "messages": [{"role": "user", "content": "..."}],  // OpenAI chat format
            "params": {},       // Optional: tools, temperature, etc.
            "metadata": {},     // Optional: generation-specific metadata
            "num_repeats": 1    // Optional: number of repetitions for each generation element
        }
    ],
    "metadata": {},             // Task-specific metadata
    "evaluation": {
        "scorer": "debunking",  // Scorer to use (maps to config)
        "data": {               // Scorer-specific evaluation criteria
            "criterion": "...",
            "context": "..."
        }
    }
}
```

> **Note:** In practice, only **biases** samples have multiple elements in the `generations` list, each representing a prompt variation to test for bias across demographic attributes.

