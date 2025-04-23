## Install notes
1. Install uv

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
- `VLLM_API_KEY` (to access VLLM models hosted on Modal)
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY` 


## Benchmark setup

To setup the benchmark, simply run:

```bash
python 01_setup_benchmark.py --save_path <path_to_save_benchmark>.db
```

The path to the files for each submodule should be set in `benchmark_config.yaml`, under the `benchmark_categories` key.
Each category should have the following structure:
```
name: <category_name>
data_path: <path_to_data>
tasks:
    - name: <task_name>
      scorer: <scorer_name>
      type: <task_type>
      description: <task_description>
```
Each task should provide a name, type, description and its associated scorer.
Path to data should point to the folder containing the jsonl files for each tasks, e.g. `../debunking/data/data_formatted_completion`, relative to the benchmark folder.

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

## Add a new category
To add a new task, follow these steps:
1. Add it in the `benchmark_config.yaml` file, with the correct `data_path` and a list of tasks. 
2. Implement the required scorers used in the tasks of the categories in the `scorers` folder and add it to the `SCORERS` inside `scorers/get_scorer.py`. 

## Add a model
To add a new model, simply add it in the `benchmark_config.yaml` file, under the `models` key. You can also change the evaluation models in the `evaluation_models` key.

## Run the benchmark
To run the benchmark, simply run:

```bash
python 02_run_benchmark.py <path_to_benchmark.db> --max_evaluations_per_task <int>
```

The `max_evaluations_per_task` argument is optional, it sets the maximum number of evaluations per task. 
