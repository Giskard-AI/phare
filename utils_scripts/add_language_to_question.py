#%%
import argparse
import json
from collections import Counter
from pathlib import Path

from lmeval import load_benchmark
from scorers.get_scorer import inject_scorers

inject_scorers()

args = argparse.ArgumentParser()
args.add_argument("data_path", type=str, help="Path to the data directory")
args.add_argument("save_path", type=str, help="Path to save the benchmark database")

args = args.parse_args()

ROOT_DIR = Path(__file__).parent.parent.parent
QUESTION_POOL_PATH = ROOT_DIR / "hallucination" / "05_data_prepared"

question_pool = []
for file in QUESTION_POOL_PATH.rglob("*.jsonl"):
    with open(file, "r") as f:
        for line in f:
            question_pool.append(json.loads(line))
language_by_id = {q["id"]: q["metadata"]["language"] for q in question_pool}

benchmark = load_benchmark(args.data_path)
benchmark.summary()

all_questions = [q for c in benchmark.categories for t in c.tasks for q in t.questions]
all_languages = [q.language for q in all_questions]
language_counts = Counter(all_languages)

print("="*100)
print(language_counts)
print("="*100)
#%%

for category in benchmark.categories:
    print(f"Category {category.name}")
    for task in category.tasks:
        questions_filtered = []
        print(f"Task {task.name} has {len(task.questions)} questions")
        for question in task.questions:
            question.language = language_by_id[question.metadata["uuid"]]

all_questions = [q for c in benchmark.categories for t in c.tasks for q in t.questions]
all_languages = [q.language for q in all_questions]
language_counts = Counter(all_languages)

print("="*100)
print(language_counts)
print("="*100)
benchmark.summary()
benchmark.save(args.save_path)
