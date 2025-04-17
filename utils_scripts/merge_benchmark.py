import argparse
import json
from lmeval import load_benchmark
from scorers.get_scorer import inject_scorers

inject_scorers()

args = argparse.ArgumentParser()
args.add_argument("base_path", type=str, help="Path to the first benchmark")
args.add_argument("merge_path", type=str, help="Path to the second benchmark")
args.add_argument("save_path", type=str, help="Path to save the merged benchmark")
args = args.parse_args()

print("== Base benchmark summary ==")
benchmark_base = load_benchmark(args.base_path)
benchmark_base.summary()

print("== To merge benchmark summary ==")
benchmark_merge = load_benchmark(args.merge_path)
benchmark_merge.summary()

def update_dict_recursively(d, u):
    """
    Recursively updates dictionary `d` with elements from dictionary `u`.
    """
    for key, value in u.items():
        if isinstance(value, dict):
            # If `d[key]` is also a dictionary, recursively update it
            if key in d and isinstance(d[key], dict):
                update_dict_recursively(d[key], value)
            else:
                d[key] = value
        else:
            d[key] = value
    return d

def get_question_id(question):
    return question.metadata.get("uuid", question.metadata.get("id"))

for category_merge in benchmark_merge.categories:
    if category_merge.name not in [
        category_base.name for category_base in benchmark_base.categories
    ]:
        benchmark_base.categories.append(category_merge)
        continue
    category_base = [
        cat for cat in benchmark_base.categories if cat.name == category_merge.name
    ][0]
    for task_merge in category_merge.tasks:
        if task_merge.name not in [task_base.name for task_base in category_base.tasks]:
            category_base.tasks.append(task_merge)
            continue
        task_base = [
            task for task in category_base.tasks if task.name == task_merge.name
        ][0]
        for question_merge in task_merge.questions:
            question_merge_id = get_question_id(question_merge)
            question_base_ids = [get_question_id(q) for q in task_base.questions]
            if question_merge_id not in question_base_ids:
                task_base.questions.append(question_merge)
                continue
            question_base = [
                q for q in task_base.questions if get_question_id(q) == question_merge_id
            ][0]
            update_dict_recursively(question_base.lm_answers, question_merge.lm_answers)

print("== Merged benchmark summary ==")
benchmark_base.summary()
benchmark_base.save(args.save_path)
