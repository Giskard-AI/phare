import argparse
import json
from lmeval import load_benchmark
from scorers.get_scorer import inject_scorers

inject_scorers()

args = argparse.ArgumentParser()
args.add_argument("data_path", type=str, help="Path to the data directory")
args = args.parse_args()

benchmark = load_benchmark(args.data_path)

benchmark.summary()


from collections import defaultdict

# Initialize counters

answers_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
evaluation_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
error_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
empty_answers_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

empty_answers_ids = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
error_ids = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Count questions per category, task and language
for category in benchmark.categories:
    for task in category.tasks:
        for question in task.questions:
            language = question.language
            answers_dict = question.lm_answers["completion_prompt-1.0"]
            for model, answer in answers_dict.items():
                if not answer.iserror:
                    choices = answer.raw_response["choices"]
                    if len(choices) == 0:
                        empty_answers_counts[category.name][task.name][model] += 1
                        empty_answers_ids[category.name][task.name][model].append(question.metadata.get("id", question.metadata.get("uuid")))
                        print("No choices found for question", question.metadata.get("id", question.metadata.get("uuid")))
                        print(f"Score: {answer.score}")
                        print(f"Model: {model}")
                        print(f"Raw response: {answer.raw_response}")
                        print("-"*100)
                    elif (answer.answer is None or answer.answer == "") and (choices[0]["message"]["tool_calls"] is None or len(choices[0]["message"]["tool_calls"]) == 0):
                        empty_answers_counts[category.name][task.name][model] += 1
                        empty_answers_ids[category.name][task.name][model].append(question.metadata.get("id", question.metadata.get("uuid")))
                        print("Empty answer for question", question.metadata.get("id", question.metadata.get("uuid")))
                        print(f"Score: {answer.score}")
                        print(f"Model: {model}")
                        print(f"Raw response: {answer.raw_response}")
                        print("-"*100)
                    else:
                        answers_counts[category.name][task.name][model] += 1
                else:
                    error_counts[category.name][task.name][model] += 1
                    question_id = question.metadata.get("id", question.metadata.get("uuid"))
                    error_ids[category.name][task.name][model].append(question_id)

# Print all dictionaries nicely
print("\n=== Answer Counts ===")
for category, tasks in answers_counts.items():
    print(f"\nCategory: {category}")
    for task, models in tasks.items():
        print(f"  Task: {task}")
        for model, count in models.items():
            print(f"    {model}: {count}")

print("\n=== Empty Answer Counts ===") 
for category, tasks in empty_answers_counts.items():
    print(f"\nCategory: {category}")
    for task, models in tasks.items():
        print(f"  Task: {task}")
        for model, count in models.items():
            print(f"    {model}: {count}")

print("\n=== Error Counts ===")
for category, tasks in error_counts.items():
    print(f"\nCategory: {category}")
    for task, models in tasks.items():
        print(f"  Task: {task}")
        for model, count in models.items():
            print(f"    {model}: {count}")

print("\n=== Error IDs ===")
for category, tasks in error_ids.items():
    print(f"\nCategory: {category}")
    for task, models in tasks.items():
        print(f"  Task: {task}")
        for model, ids in models.items():
            print(f"    {model}: {ids}")


print("\n=== Empty Answer IDs ===")
for category, tasks in empty_answers_ids.items():
    print(f"\nCategory: {category}")
    for task, models in tasks.items():
        print(f"  Task: {task}")
        for model, ids in models.items():
            print(f"    {model}: {ids}")
