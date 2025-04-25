import json
import os
import numpy as np
import yaml
import argparse
import shutil

import huggingface_hub

from lmeval import Question, Task, TaskType, load_benchmark, GroupedQuestion
from lmeval import Category, Benchmark, QuestionSource
from lmeval.models.litellm import LiteLLMModel
from scorers.get_scorer import get_scorer
from pathlib import Path
from scorers.majority_vote_model import MajorityVoteEvaluationModel
from scorers.get_scorer import inject_scorers

inject_scorers()
args = argparse.ArgumentParser()
args.add_argument(
    "--config_path",
    type=str,
    help="Path to the benchmark config file",
    default=Path(__file__).parent / "benchmark_config.yaml",
)
args.add_argument(
    "--existing_benchmark",
    type=str,
    default=None,
    help="Path to the existing benchmark database",
)
args.add_argument(
    "--save_path",
    type=str,
    default="results/full_demo_benchmark.db",
    help="Path to save the benchmark database",
)
args.add_argument("--seed", type=int, default=1729, help="Seed for random sampling")
args = args.parse_args()

SEED = args.seed
SAVE_PATH = args.save_path
BENCHMARK_CONFIG = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)

CURRENT_PATH = Path(__file__).parent
DATA_PATH = CURRENT_PATH / "data"

rng = np.random.default_rng(SEED)

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)

# create a benchmark object
is_benchmark_update = False
if args.existing_benchmark:
    benchmark = load_benchmark(args.existing_benchmark)
    is_benchmark_update = True
else:
    benchmark = Benchmark(
        name=BENCHMARK_CONFIG["benchmark_name"],
        description=BENCHMARK_CONFIG["benchmark_description"],
    )

evaluation_model = MajorityVoteEvaluationModel(
    models=[
        LiteLLMModel(
            litellm_model=eval_model["litellm_model"],
            model_version=eval_model["model_version"],
            publisher=eval_model["publisher"],
        )
        for eval_model in BENCHMARK_CONFIG["evaluation_models"]
    ],
    weights=[
        eval_model["weight"] for eval_model in BENCHMARK_CONFIG["evaluation_models"]
    ],
)


def parse_question(question_dict, question_source) -> Question:
    question_lang = (
        question_dict["metadata"]["language"]
        if "metadata" in question_dict and "language" in question_dict["metadata"]
        else "en"
    )
    evaluation_data = question_dict["evaluation_data"]
    evaluation_data.update({"uuid": question_dict["id"]})

    if "task" not in evaluation_data and "task_name" not in evaluation_data:
        evaluation_data["task_name"] = question_dict["metadata"].get("task", question_dict["metadata"].get("task_name"))

    tools = question_dict.get("tools", None)
    if tools is not None and len(tools) == 0:
        tools = None

    question = Question(
        language=question_lang,
        messages=question_dict["messages"],
        question=(
            evaluation_data["question"]
            if "question" in evaluation_data
            else question_dict["messages"][-1]["content"]
        ),
        source=question_source,
        metadata=evaluation_data,
        tools=tools,
    )
    return question


categories = BENCHMARK_CONFIG["benchmark_categories"]
tasks = {}

# Prepare the data paths
path_dict = {}
failed_path_dict = {}
for category in categories:
    for task_config in category["tasks"]:
        try:
            full_path = f"{category['data_path']}/{task_config['name']}.jsonl"
            temp_path = huggingface_hub.hf_hub_download(
                repo_id=category["hf_dataset"],
                repo_type="dataset",
                filename=full_path
            )
            path_dict.update({
                task_config["name"]: {
                    "raw_path": full_path,
                    "processed_path": temp_path,
                },
            })
        except Exception as e:
            print(f"Failed to download {task_config['name']}: {e}")
            failed_path_dict.update({
                task_config["name"]: {
                    "raw_path": full_path,
                    "processed_path": None,
                },
            })
            continue

# Stop if any paths failed to download
if len(failed_path_dict) > 0:
    print("Some paths failed to download:")
    for name, paths in failed_path_dict.items():
        print(f"\tFailed to download '{name}': {paths['raw_path']}")
    print("Please check the paths and try again. Exiting...")
    exit(1)


for category in categories:
    category_questions = []
    if is_benchmark_update:
        benchmark_category = benchmark.get_category(category["name"])
    else:
        benchmark_category = Category(name=category["name"])
        benchmark.add_category(benchmark_category)

    for task_config in category["tasks"]:
        if is_benchmark_update:
            task = benchmark_category.get_task(task_config["name"])
            category_questions.extend(
                [question.metadata["uuid"] for question in task.questions]
            )
        else:
            task_scorer = get_scorer(task_config["scorer"])
            task_type = TaskType[task_config["type"]]
            task_scorer.model = evaluation_model

            task = Task(
                name=task_config["name"], type=task_type, scorer=task_scorer
            )
            benchmark_category.add_task(task)
        task_id = f"{category['name']}/{task_config['name'].split('/')[-1]}"
        tasks[task_id] = task

        question_jsonl = path_dict[task_config["name"]]["processed_path"]

        if not question_jsonl.endswith(".jsonl"):
            print(f"Skipping '{question_jsonl}' as it is not a JSONL file.")
            continue

        question_source = QuestionSource(
            name=".".join(question_jsonl.split(".")[:-1])
        )

        with open(question_jsonl, "r") as f:
            lines = f.readlines()
            rng.shuffle(lines)

            for line in filter(lambda x: x.strip() != "", lines):
                # Load sample data
                data = json.loads(line)

                # If data contains a set of questions, we parse as a GroupedQuestion
                if "question_set" in data:
                    num_repeats = data["metadata"].get("num_repeats", 1)
                    print(f"num_repeats: {num_repeats}")
                    question_set = [
                        parse_question(q, question_source)
                        for q in data["question_set"]
                    ] * num_repeats
                    question_to_add = GroupedQuestion(
                        question_set=question_set,
                        metadata={"id": data["id"], **data["metadata"]},
                        source=question_source,
                    )
                # Otherwise, we use the standard Question class
                else:
                    question_to_add = parse_question(data, question_source)

                question_id = question_to_add.metadata.get(
                    "id", question_to_add.metadata.get("uuid")
                )

                question_task = question_to_add.metadata.get(
                    "task_name", question_to_add.metadata.get("task")
                )
                if question_id not in category_questions:
                    tasks[task_id].add_question(question_to_add)

benchmark.summary()
benchmark.save(SAVE_PATH)
