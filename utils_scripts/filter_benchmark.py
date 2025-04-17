import argparse
import json
from lmeval import load_benchmark
from scorers.get_scorer import inject_scorers

inject_scorers()

args = argparse.ArgumentParser()
args.add_argument("data_path", type=str, help="Path to the data directory")
args.add_argument("save_path", type=str, help="Path to save the benchmark database")
args.add_argument(
    "id_file", type=str, help="Path to the file containing the ids to filter"
)
args = args.parse_args()

benchmark = load_benchmark(args.data_path)
benchmark.summary()

IDS_TO_FILTER = []

with open(args.id_file, "r") as f:
    for line in f:
        IDS_TO_FILTER.append(line.strip())

for category in benchmark.categories:
    print(f"Category {category.name}")
    for task in category.tasks:
        questions_filtered = []
        print(f"Task {task.name} has {len(task.questions)} questions")
        for question in task.questions:
            if question.metadata["uuid"] not in IDS_TO_FILTER:
                questions_filtered.append(question)
        task.questions = questions_filtered
        print(f"Keeping {len(task.questions)} questions for the task.")

benchmark.summary()
benchmark.save(args.save_path)
