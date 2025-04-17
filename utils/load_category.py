import pandas as pd
import json
from pathlib import Path
from functools import reduce
import argparse

from lmeval import load_benchmark
from scorers.get_scorer import inject_scorers

inject_scorers()


ROOT_PATH = Path(__file__).parent.parent.parent
SPLIT_IDS_PATH = ROOT_PATH / "samples" / "splits_ids.json"

PUBLIC_SPLIT_IDS = json.load(open(SPLIT_IDS_PATH))["public"]
PRIVATE_SPLIT_IDS = json.load(open(SPLIT_IDS_PATH))["private"]

heatmap_variable_mapping = {
    "debunking": "args/prompt_level",
    "misinformation": "template_type",
    "factuality": "template_type",
    "tools_usage": "perturbation_type",
    "harmful_vulnerable_misguidance": "category",
}

def deep_get(dictionary, *keys, default=None):
    return reduce(
        lambda d, key: d.get(key, default) if d and isinstance(d, dict) else default,
        keys,
        dictionary,
    )


def load_category_data(category):
    # Create a subdirectory for the module
    flattened = []
    heatmap_variable_path = heatmap_variable_mapping[category.name]
    heatmap_variable_name = heatmap_variable_path.split("/")[-1]
    for task in category.tasks:
        for question in task.questions:
            language = question.language
            task_name = task.name
            question_id = question.metadata.get("id", question.metadata.get("uuid"))

            heatmap_variable = deep_get(question.metadata, *heatmap_variable_path.split("/"), default=None)
            if "completion_prompt-1.0" in question.lm_answers:
                for model in question.lm_answers["completion_prompt-1.0"]:
                    answer = question.lm_answers["completion_prompt-1.0"][model]
                    if answer.iserror:
                        continue
                    if len(answer.raw_response["choices"]) == 0:
                        continue
                    elif (answer.answer is None or answer.answer == "") and (
                        answer.raw_response["choices"][0]["message"]["tool_calls"]
                        is None
                        or len(
                            answer.raw_response["choices"][0]["message"]["tool_calls"]
                        )
                        == 0
                    ):
                        continue

                    row = {
                        "model": model,
                        "category": category.name,
                        "task": task_name,
                        "language": language,
                        "task_language": f"{task_name}_{language}",
                        "score": answer.score,
                        "split": "public"
                            if question_id in PUBLIC_SPLIT_IDS
                            else "private",
                        "heatmap_variable": heatmap_variable,
                        "heatmap_variable_name": heatmap_variable_name,
                        "evaluation_raw": answer.raw_response

                    }
                    flattened.append(row)

    df = pd.DataFrame(flattened)
    return df


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("benchmark", type=str)
    args = args.parse_args()

    benchmark_path = Path(args.benchmark)

    benchmark = load_benchmark(str(benchmark_path))
    df_list = []
    for category in benchmark.categories:
        df = load_category_data(category)
        df_list.append(df)

    df = pd.concat(df_list)
    df["heatmap_variable"] = df["heatmap_variable"].astype(str)
    df.to_parquet(benchmark_path.parent / f"{benchmark_path.stem}.parquet", index=False)
    print(f"Saved to {benchmark_path.parent / f'{benchmark_path.stem}.parquet'}")
    print(f"Number of rows: {len(df)}")