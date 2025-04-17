from lmeval.scorers.loader import add_scorer
from .factuality_scorer import FactualityScorer
from .misinformation_scorer import SatiricalQuestionScorer
from .debunking_scorer import DebunkingScorer
from .tools_scorer import ToolsScorer
from .biases_scorer import BiasesScorer
from .harmful_scorer import HarmfulMisguidanceScorer

SCORERS = {
    "factuality": FactualityScorer,
    "misinformation": SatiricalQuestionScorer,
    "debunking": DebunkingScorer,
    "tools": ToolsScorer,
    "biases_story_generation": BiasesScorer,
    "harmful_misguidance": HarmfulMisguidanceScorer,
}


# ugly but we need to add the scorers in the loader to use the benchmark loading function
def inject_scorers():
    for k in SCORERS:
        add_scorer(k, SCORERS[k])


def get_scorer(scorer_name: str):
    return SCORERS[scorer_name]()
