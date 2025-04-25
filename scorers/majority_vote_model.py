import json
import regex
from typing import Type
from pydantic import BaseModel

from lmeval.custom_model import CustomModel
from lmeval.models.lmmodel import LMModel


def extract_json_object(response: str) -> dict:
    """
    Extract the json object from the response.
    """
    pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    try:
        return json.loads(pattern.findall(response)[0])
    except (json.JSONDecodeError, IndexError) as e:
        return None

class VoteException(Exception):
    def __init__(self, message):
        super().__init__(message)


class MajorityVote(CustomModel):
    decision: bool
    raw_responses: dict[str, dict]


class MajorityVoteEvaluationModel(LMModel):
    models: list[LMModel]
    weights: list[float]

    def majority_vote(
        self,
        messages: list[dict],
        decision_key: str,
        response_format: Type[BaseModel],
        **generation_kwargs,
    ) -> MajorityVote:
        votes = {}

        for model, weight in zip(self.models, self.weights):
            try:
                response = model.complete(
                    messages,
                    response_format=response_format,
                    **generation_kwargs,
                )

                response = extract_json_object(response.answer)

                votes[model.name] = {
                    "response": response[decision_key],
                    "weight": weight,
                }
                if "reason" in response:
                    votes[model.name]["reason"] = response["reason"]
            except Exception as exception:
                print(response)
                print(
                    "Error in evaluation with model", model.name, ":", repr(exception)
                )
                continue

        total_weight = sum(self.weights)

        pass_weight_sum = sum(
            vote["weight"] for vote in votes.values() if vote["response"] == 1
        )
        fail_weight_sum = sum(
            vote["weight"] for vote in votes.values() if vote["response"] == 0
        )

        # Check for consensus
        if pass_weight_sum > total_weight / 2:
            return MajorityVote(decision=True, raw_responses=votes)
        elif fail_weight_sum > total_weight / 2:
            return MajorityVote(decision=False, raw_responses=votes)
        else:
            print(votes)
            raise VoteException("No consensus reached")
