# Phare Benchmark

Phare is a multilingual benchmark that measures LLM Safety across multiple categories of vulnerabilities, including **hallucination**, **biases & stereotypes** and **harmful content**.

![Phare Benchmark](.github/phare.png)

## Background and Motivation

Large Language Models (LLMs) have rapidly advanced in capability and adoption, becoming essential tools across a wide spectrum of natural language processing applications. While existing benchmarks have focused primarily on general performance metrics, such as accuracy and task completion, there is a growing recognition of the need to evaluate these models through the lens of safety and robustness. Concerns over hallucination, harmful outputs, and social bias have escalated alongside model deployment in sensitive and high-impact settings.

However, current safety evaluations tend to be fragmented or limited in scope, lacking unified diagnostic tools that deeply probe model behavior. In response to this gap, we introduce Phare, a multilingual and multi-faceted evaluation framework designed specifically to diagnose and analyze LLM failure modes across hallucination, bias, and harmful content. Phare aims at contributing a tool for the development of safer and more trustworthy AI systems.

## Usage

Phare is easy to use and reproducible. You can set up and run the benchmark with just a few commands using the `uv` package manager.

### Install notes
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Clone this repo:
```bash
git clone https://github.com/Giskard-AI/phare
```

3. Install the requirements:
```bash
uv sync
source .venv/bin/activate
```

4. Setup secrets:
Running the benchmark will requires tokens for calling the different models. Here is a list of expected env variables: 
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY` 

### Benchmark setup

To setup the benchmark, simply run:

```bash
python 01_setup_benchmark.py --config_path <path_to_config>.yaml --save_path <path_to_save_benchmark>.db
```

The Hugging Face repository and the path to the files for each submodule should be set in `benchmark_config.yaml`, under the `hf_dataset` and the `data_path` keys.
Each category should have the following structure:
```
name: <category_name>
hf_dataset: giskardai/phare
data_path: <path_to_data>
tasks:
    - name: <task_name>
      scorer: <scorer_name>
      type: <task_type>
      description: <task_description>
```
Each task should provide a name, type, description and its associated scorer.
Path to data should point to the folder under the Hugging Face repository, containing the jsonl files for each tasks.

For example, in the `giskardai/phare` repository, the `hallucination/debunking` as the `<path_to_data>` with `misconceptions` as `<task_name>` indicates the `hallucination/debunking/misconceptions.jsonl` [file](https://huggingface.co/datasets/giskardai/phare/blob/main/hallucination/debunking/misconceptions.jsonl).

Inside the jsonl files, each line should have the following format: 

```json
{
    "id": "question_uuid",
    "messages": [{"role": "user", "content": "..."}, ...],
    "metadata": {
        "task": "category_name/task_name",
        "language": "en",

    },
    "evaluation_data": {
        ...
    }
}
```

## Run the benchmark
To run the benchmark, simply run:

```bash
python 02_run_benchmark.py <path_to_benchmark.db> --max_evaluations_per_task <int>
```

The `max_evaluations_per_task` argument is optional, it sets the maximum number of evaluations per task. 

## Extensibility

This Phare benchmark is also designed to be easily extensible, thanks to [LM-eval](https://github.com/google/lmeval). You can add new categories, tasks, and models by following the instructions below.

### Add a new category
To add a new task, follow these steps:
1. Add it in the `benchmark_config.yaml` file, with the correct `data_path` and a list of tasks. 
2. Implement the required scorers used in the tasks of the categories in the `scorers` folder and add it to the `SCORERS` inside `scorers/get_scorer.py`. 

### Add a model
To add a new model, simply add it in the `benchmark_config.yaml` file, under the `models` key. You can also change the evaluation models in the `evaluation_models` key.
