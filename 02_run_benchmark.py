import argparse
from datetime import datetime
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

from lmeval import load_benchmark
from lmeval.models.litellm import LiteLLMModel
from lmeval.evaluator import Evaluator
from lmeval import set_log_level
from scorers.prompt import CompletionPrompt, GroupedCompletionPrompt

from scorers.get_scorer import inject_scorers
from scorers.majority_vote_model import MajorityVoteEvaluationModel


load_dotenv()
inject_scorers()
set_log_level("CRITICAL")

args = argparse.ArgumentParser()
args.add_argument("benchmark_path", type=str)
args.add_argument("--max_evaluations_per_task", type=int, default=10000)

args.add_argument(
    "--config_path",
    type=str,
    help="Path to the benchmark config file",
    default=Path(__file__).parent / "benchmark_config.yaml",
)
args.add_argument("--debug", action="store_true")

args = args.parse_args()

BENCHMARK_CONFIG = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
MODELS_LIST = BENCHMARK_CONFIG["models"]

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

benchmark = load_benchmark(args.benchmark_path)
for category in benchmark.categories:
    for task in category.tasks:
        task.scorer.model = evaluation_model
        task.scorer.debug = args.debug

# models to evaluate
models = []
if len(MODELS_LIST) > 0:
    for model in MODELS_LIST:
        generation_kwargs = model.get("generation_kwargs", {})

        api_key = None
        base_url = None
        if "vllm" in generation_kwargs and generation_kwargs.pop("vllm"):
            api_key = os.getenv("VLLM_API_KEY")
            base_url = generation_kwargs.pop("base_url")

        model = LiteLLMModel(
            litellm_model=model["litellm_model"],
            publisher=model["publisher"],
            model_version=model["name"],
            api_key=api_key,
            base_url=base_url,
        )
        model.runtime_vars["generation_kwargs"] = generation_kwargs
        model.runtime_vars["supports_system_prompt"] = generation_kwargs.pop(
            "supports_system_prompt", True
        )
        models.append(model)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_path = (
        Path(args.benchmark_path).parent / f"checkpoint_run_{timestamp_str}.db"
    )
    print(f"Checkpoint will be saved to {checkpoint_path}")
    evaluator = Evaluator(benchmark, save_path=str(checkpoint_path))
    eval_plan = evaluator.plan(
        models,
        [CompletionPrompt(), GroupedCompletionPrompt()],
        max_evaluations_per_task=args.max_evaluations_per_task,
    )  # plan evaluation
    completed_benchmark = evaluator.execute()  # run evaluation

    benchmark.summary()
    benchmark.save(args.benchmark_path)

    print(f"Benchmark saved to {args.benchmark_path}")
