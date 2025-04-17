import argparse
import json
from lmeval import load_benchmark
from scorers.get_scorer import inject_scorers

inject_scorers()

args = argparse.ArgumentParser()
args.add_argument("data_path", type=str, help="Path to the data directory")
args.add_argument("save_path", type=str, help="Path to save the benchmark database")
args = args.parse_args()

benchmark = load_benchmark(args.data_path)

benchmark_dump = benchmark.model_dump()


for category in benchmark_dump["categories"]:
    for task in category["tasks"]:
        task["modality"] = str(task["modality"].value)
        task["scorer"]["type"] = str(task["scorer"]["type"])
        task["scorer"]["modality"] = str(task["scorer"]["modality"].value)
        task["level"] = str(task["level"].value)


with open(args.save_path, "w") as f:
    json.dump(benchmark_dump, f)
