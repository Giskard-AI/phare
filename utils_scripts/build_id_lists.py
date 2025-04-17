#%%
import json
import os
import glob
from pathlib import Path
from collections import defaultdict
DATA_PATH = Path(__file__).parent.parent.parent / "samples"

splits = ["public", "private"]


ids_benchmark = defaultdict(list)
ids_benchmark_template_type = defaultdict(lambda: defaultdict(list))
for split in splits:
    for submodule in os.listdir(DATA_PATH / f"{split}_set"):
        for task in os.listdir(DATA_PATH / f"{split}_set" / submodule):
            for file in glob.glob(
                str(DATA_PATH / f"{split}_set" / submodule / task / "*.jsonl")
            ):
                with open(file, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        question_id = data["id"]
                        ids_benchmark[split].append(question_id)
                        ids_benchmark[f"{split}_{task}"].append(question_id)
                        if "question_set" in data:
                            for question in data["question_set"]:
                                ids_benchmark[f"{split}_biases_sub_questions"].append(question["id"])
                        
                        if task in ["factuality", "satirical"]:
                            ids_benchmark_template_type[f"{split}_{task}"][data["id"]].append(data["evaluation_data"]["template_type"])
#%%

for k, v in ids_benchmark.items():
    print(k, len(v), len(set(v)))

print("-" * 100)
for k, v in ids_benchmark_template_type.items():
    print(k, len(v))
# %%
json.dump(ids_benchmark, open(DATA_PATH / "splits_ids.json", "w"))
# %%
