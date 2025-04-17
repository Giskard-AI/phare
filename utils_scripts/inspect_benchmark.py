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

N_SAMPLES_PER_TASK = 25

# VARIABLE_TO_INSPECT = ["reference_answer"]
VARIABLE_TO_INSPECT = ["perturbation_type", "api_call", "template_type"]
MODEL = "Gemini 2.0 Flash"

for category in benchmark.categories:
    for task in category.tasks:
        print(task.name)
        print("=" * 100)
        for question in task.questions[:N_SAMPLES_PER_TASK]:
            # print(question.lm_answers["completion_prompt-1.0"])
            if "completion_prompt-1.0" not in question.lm_answers:
                print(f"Question: {question.question}")
                print(f"Tools: {question.tools}")
                for var in VARIABLE_TO_INSPECT:
                    print(f"{var}: {question.metadata[var]}")
                print("-" * 100)
            else:
                for model in question.lm_answers["completion_prompt-1.0"]:
                    if model != MODEL:
                        continue
                    print(f"Question: {question.question}")
                    print(f"Answer: {question.lm_answers['completion_prompt-1.0'][model].answer}")
                    print(f"Score: {question.lm_answers['completion_prompt-1.0'][model].score}")
                    print(f"Raw response: {question.lm_answers['completion_prompt-1.0'][model].raw_response}")
                    print(f"Tools: {question.tools}")
                    print(
                        f"Evaluation: {question.lm_answers['completion_prompt-1.0'][model].score_raw_data}"
                    )
                   
                    print("-" * 100)
