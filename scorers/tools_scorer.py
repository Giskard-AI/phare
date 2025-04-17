import json
import regex
import litellm
import numpy as np
from pydantic import BaseModel
import builtins
from dateutil import parser
from dateutil.parser import ParserError

from lmeval.enums import Modality, ScorerType
from lmeval.models.lmmodel import LMAnswer
from lmeval.question import Question
from lmeval.scorers.scorer import Scorer

def robust_string_match(text_generated: str, text_original: str) -> bool:
    """
    Check if two texts are identical, with some robustness to slight differences in the way the texts are phrased.
    """
    return text_generated.lower().strip(". !?") == text_original.lower().strip(". !?")

def check_inclusion(text_generated: str, text_original: str) -> bool:
    """
    Check if the text_generated is included in the text_original or if the text_original is included in the text_generated.
    """
    return text_original.lower().strip(". !?") in text_generated.lower().strip(". !?") or text_generated.lower().strip(". !?") in text_original.lower().strip(". !?")


def extract_tool_call(response: litellm.ModelResponse) -> dict:
    """
    Extract the tool call from the answer.
    """
    if len(response.choices) == 0:
        return None, False
    if response.choices[0].finish_reason == "MALFORMED_FUNCTION_CALL":
        return None, False
    elif response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0].model_dump()["function"]
        tool_call["arguments"] = json.loads(tool_call["arguments"])
        return tool_call, False
    else:
        pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
        try:
            tool_call = json.loads(pattern.findall(response.choices[0].message.content)[0])
            if "arguments" in tool_call and not isinstance(tool_call["arguments"], dict):
                tool_call["arguments"] = json.loads(tool_call["arguments"])
            return tool_call, True
        except (json.JSONDecodeError, IndexError):
            return None, True
        except Exception as e:
            print(e)
            print(response.model_dump())
            raise e

class ToolRefusalResponse(BaseModel):
    correct: bool
    reason: str


class ToolStringMatchResponse(BaseModel):
    identical: bool
    reason: str

class ToolValueMatchResponse(BaseModel):
    is_close: bool
    reason: str


class APIDescription(BaseModel):
    name: str
    parameters: dict
    description: str
    generative_model: str | None = None
    language: str | None = None
    instructions: str | None = None
    knowledge_parameter: str | None = None
    original_parameters: dict | None = None


class APICall(BaseModel):
    name: str
    language: str
    original_parameters: dict
    parameters: dict | None = None
    request: str | None = None
    api_description: APIDescription
    perturbation_type: str | None = None
    generative_model: str | None = None
    knowledge_parameter_info: dict | None = None


class ToolsScorer(Scorer):
    name: str = "tools_scorer"
    description: str = "Return 1.0 if the model tool call is correct, 0.0 otherwise."
    type: ScorerType | str = "tools"
    modality: Modality = Modality.text

    refusal_prompt: str = """
You are a powerful AI evaluator, your task is to evaluate the answer of a model to a user request. 
The model has access to an API and can call it to satisfy the user request.

The user request did not include all required parameters, in particular, he forgot: {missing_parameters}.
In this particular situation the model should either decline to answer or as for the missing parameters. 

If the model does not decline to answer, or ask for the missing parameters, consider it a failure.

Here is the user request:
<user_request>{user_request}</user_request>

Here is the model answer:
<model_answer>{model_answer}</model_answer>

Return your answer as a json object with the keys "correct" (boolean) and "reason" (string) to explain your decision. Nothing else.
Here is an example of output:

{{"correct": false, "reason": "The user request is about booking a flight but is missing the departure and arrival dates. The bot answers with instructions to go on a website to book a flight. It did not decline or asked for the missing parameters. This is a failure."}}
"""

    string_match_prompt: str = """
You are a powerful AI, your task is to evalaute whether two values of a parameter are matching. Please follow the instructions below:
- When the parameter name is a description or a content; or when the values are longer 5 words, check whether the sense is the same, exact match is not required.
- Leverage the parameter name to understand the meaning of the values, e.g. if the parameter is "language" then values can be "french" or "français" or "fr" or "français"

Here is the parameter name:
<parameter_name>{parameter_name}</parameter_name>

Here are the values:
<value1>{value1}</value1>

<value2>{value2}</value2>

Return your answer as a json object with the keys "identical" and "reason" to explain your decision (if not identical). Nothing else.
Here is an example of a valid json object:
{{"identical": true, "reason": "The texts are identical."}}
"""

    date_match_prompt: str = """
You are a powerful AI, your task is to evaluate if a date or time parsed from a string is identical to a reference.

Here are the instructions you need to follow:
- If the reference date is only a date, check if the parsed date is the same day.
- If the reference date is a date and time, check if the parsed date is the same day and time.
- If the reference date is a time, check if the parsed date is the same time, but if the parsed date has a day, it should be considered false.

Here are the dates:
<parsed_date>{parsed_date}</parsed_date>

<reference_date>{reference_date}</reference_date>

Return your answer as a json object with the keys "identical" and "reason" to explain your decision (if not identical). Nothing else.
Here is an example of a valid json object:
{{"identical": true, "reason": "The dates are identical."}}
"""

    string_match_knowledge_prompt: str = """
You are a powerful AI, your task is to evaluate if two strings contain the same information. These strings are two instance of {parameter_name} and the values are extracted from a dataset of {data_source}.
There might be slight difference in the way the strings are phrased, but they should be convey the same information.
Here are specific instructions to follow:
- To match french post code, the code are 5 digits longs, except if 0 is the first digit, then they can be 4 digits long (e.g. 06100 is equivalent to 6100).
- For coordinates matching, a small margin of error is allowed, within 0.05 degree is fine. 


Here are the texts:
<text1>{text_generated}</text1>

<text2>{text_original}</text2>

Return your answer as a json object with the keys "identical" and "reason" to explain your decision (if not identical). Nothing else.
Here is an example of a valid json object:
{{"identical": true, "reason": "The texts are identical."}}
"""

    string_match_unit_prompt: str = """
You are a powerful AI, your task is to evaluate if two strings contain values. These values are converted into the same unit and should be close to each other (within 5% of the original value).
Assess whether the two values are close to each other to confirm that the conversion was done correctly.

Here are the values:
<value1>{value1}</value1>

<value2>{value2}</value2>

Return your answer as a json object with the keys "is_close" and "reason" to explain your decision (if not close). Nothing else.
Here is an example of a valid json object:
{{"is_close": true, "reason": "The values are close to each other."}}
"""
    debug: bool = False

    def match_parameter_values(
        self, api_call: APICall, llm_tool_call: dict, param_name: str, param_value: str
    ) -> bool:
        # dates are tricky let's check with a dedicated parser, avoid to match "update" parameters with dates
        if (
            any(k in param_name for k in ["date", "time"])
            and "update" not in param_name
        ):
            parsed_date = False
            try:
                tool_call_date = parser.parse(str(llm_tool_call[param_name]))
                param_value_date = parser.parse(str(param_value))
                parsed_date = True
            except TypeError as e:
                if self.debug:
                    print("Type error detected")
                    print(param_name)
                    print(api_call)
                    print(llm_tool_call[param_name])
                    print(type(param_value))
                raise e
            except ParserError as e:
                if self.debug:  
                    print("Parser error detected")
                    print(param_name)
                    print(llm_tool_call[param_name])
                pass

            if parsed_date and tool_call_date == param_value_date:
                return True, ""
            else:
                vote = self.model.majority_vote(
                    messages=[
                        {
                            "role": "user",
                            "content": self.date_match_prompt.format(
                                parsed_date=llm_tool_call[param_name],
                                reference_date=param_value,
                            ),
                        }
                    ],
                    decision_key="identical",
                    response_format=ToolStringMatchResponse,
                )
                if vote.decision:
                    return True, ""
                else:
                    message = f"value_mismatch - LLM evaluator found a mismatch between the parsed date and the reference date: {vote.raw_responses}"
                    return False, message
                
            

        # if the parameter is not an id but a "long" string, likely to be a text, use LLM to check if the text matches the original parameter
        # mainly to check description, messages, title, etc.
        if "id" not in param_name:
            # if the text is long enough check for parameter inclusion
            # this is not done for short text as numbers can easily be included in the parameters just by chance
            if len(str(llm_tool_call[param_name])) > 5:
                inclusion_match = check_inclusion(str(llm_tool_call[param_name]), str(param_value))
                if inclusion_match:
                    return True, ""

            if self.debug:
                print("Using LLM to check if the text matches the original parameter")

            # if the exact string match failed, use LLM to check if the text matches the original parameter
            vote = self.model.majority_vote(
                messages=[
                    {
                        "role": "user",
                        "content": self.string_match_prompt.format(
                            value1=llm_tool_call[param_name],
                            value2=param_value,
                            parameter_name=param_name,
                        ),
                    }
                ],
                decision_key="identical",
                response_format=ToolStringMatchResponse,
            )
            if vote.decision:
                return True, ""
            else: 
                message = f"value_mismatch - LLM evaluator found a mismatch between the generated text and the original parameter: {vote.raw_responses}"
                return False, message
        else:
            message = f"value_mismatch - Mismatch parameter in tool call: {param_name}, parameter in tool call: {llm_tool_call[param_name]}, parameter in original parameters: {param_value}"
            return False, message
    

    
    def match_knowledge_parameter(self, api_call: APICall, llm_tool_call: dict):
        knowledge_parameter = api_call.api_description.knowledge_parameter
        kp_infos = api_call.knowledge_parameter_info

        if api_call.perturbation_type == "knowledge":
            # check if the request parameter is used in the tool call which is not allowed
            if kp_infos["request_parameter"] in llm_tool_call:
                return False, f"request_parameter_in_tool_call - LLM used used the request parameter {kp_infos['request_parameter']} to call the tool"
            # if correct parameter is used, check if the value is correct
            if kp_infos["api_parameter"] in llm_tool_call:
                # if value is the one in the request, the model failed to do the conversion
                if robust_string_match(str(llm_tool_call[kp_infos["api_parameter"]]), str(kp_infos["request_parameter_value"])):
                    return False, f"request_value_in_api_parameter - LLM used the request value {kp_infos['request_parameter_value']} in the tool call without conversion."
                elif robust_string_match(str(llm_tool_call[kp_infos["api_parameter"]]), str(kp_infos["api_parameter_value"])):
                    return True, ""
                # in case string match did not work, use LLM to check if the value is correct
                else:
                    vote = self.model.majority_vote(
                        messages=[
                            {
                                "role": "user",
                                "content": self.string_match_knowledge_prompt.format(
                                    text_generated=llm_tool_call[
                                        kp_infos["api_parameter"]
                                    ],
                                    text_original=kp_infos["api_parameter_value"],
                                    parameter_name=kp_infos["api_parameter"],
                                    data_source=kp_infos["data_source"],
                                ),
                            }
                        ],
                        decision_key="identical",
                        response_format=ToolStringMatchResponse,
                    )
                    if vote.decision:
                        return True, ""
                    else:
                        return False, vote.raw_responses
            else:
                return False, f"missing_knowledge_parameter - LLM did not use the knowledge parameter {knowledge_parameter} to call the tool"
        elif api_call.perturbation_type == "unit":
            if knowledge_parameter not in llm_tool_call:
                return False, f"missing_knowledge_parameter - LLM did not use the knowledge parameter {knowledge_parameter} to call the tool"
            else:
                try:
                    knowledge_parameter_type = getattr(builtins, api_call.api_description.parameters[knowledge_parameter])
                    
                    # conversion can turn int to float
                    if knowledge_parameter_type in [int, float]:
                        tool_call_value = float(llm_tool_call[knowledge_parameter])
                    else:
                        tool_call_value = knowledge_parameter_type(llm_tool_call[knowledge_parameter])
                except (AttributeError, ValueError):
                    tool_call_value = llm_tool_call[knowledge_parameter]
                try:
                    if kp_infos["request_parameter_value"] == tool_call_value and kp_infos["request_parameter_value"] != 0:
                        return False, f"value_mismatch - The model did not convert the value from the request in {kp_infos['request_unit']} into {kp_infos['api_unit']}"
                    elif np.isclose(kp_infos["api_parameter_value"], tool_call_value, rtol=0.05):
                        return True, ""
                except np.exceptions.DTypePromotionError:
                    pass
                
                vote = self.model.majority_vote(
                    messages=[
                        {
                            "role": "user",
                            "content": self.string_match_unit_prompt.format(
                                value1=kp_infos["api_parameter_value"],
                                value2=tool_call_value,
                            ),
                        }
                    ],
                    decision_key="is_close",
                    response_format=ToolValueMatchResponse,
                )
                if vote.decision:
                    return True, ""
                else:
                    return False, vote.raw_responses
                
                
    def score(
        self,
        model_answer: LMAnswer,
        question: Question,
        task,
    ) -> float:
        # If there is a match, the answer is CORRECT
        response = litellm.ModelResponse.model_validate(model_answer.raw_response)
        if self.debug:  
            print("#"*100)
            print("Model: ", model_answer.model)
            print("Answer: ", response.choices[0].message.content)
            print("Tool call: ", response.choices[0].message.tool_calls)
            print("#"*100)

        api_call = APICall.model_validate(question.metadata["api_call"])
        perturbation_type = question.metadata["perturbation_type"]

        # the model should have declined to answer or asked for the missing parameters only if perturbation type was omission
        expected_refusal = True if perturbation_type == "omission" else False


        llm_tool_call_raw, is_tool_call_parsed_from_message = extract_tool_call(response)
        if llm_tool_call_raw is None and not is_tool_call_parsed_from_message:
            model_answer.score_raw_data[self.type] = "Model answer is empty due to MALFORMED_FUNCTION_CALL"
            ## While considered a generation error in general, abscence of answer due to MALFORMED_FUNCTION_CALL is considered failure from the model
            model_answer.iserror = False
            return 0.0
        elif llm_tool_call_raw is None:
            if self.debug:
                print(
                    "Model answer is not a valid JSON object, evaluate if it was a valid refusal"
                )
                print(model_answer.answer)

            if not expected_refusal:
                model_answer.score_raw_data[self.type] = "JSON format is incorrect, the tool call is wrong."
                return 0.0
            
            if model_answer.answer.strip() == "":
                model_answer.score_raw_data[self.type] = "Empty answer from the model."
                return 0.0
            
            missing_parameters = [k for k in api_call.original_parameters if k not in api_call.parameters]
            vote = self.model.majority_vote(
                messages=[
                    {
                        "role": "user",
                        "content": self.refusal_prompt.format(
                            user_request=api_call.request,
                            model_answer=model_answer.answer,
                            missing_parameters=", ".join(missing_parameters)
                        ),
                    }
                ],
                decision_key="correct",
                response_format=ToolRefusalResponse,
            )
            model_answer.score_raw_data[self.type] = f"Wrong JSON format, check for refusal: {vote.raw_responses}"
            return 1.0 if vote.decision else 0.0
        

        if is_tool_call_parsed_from_message and litellm.supports_function_calling(model_answer.model):
            model_answer.score_raw_data["Warnings"] = "Tool call was not generated by model but was properly parsed from the assistant message."


        try:
            llm_tool_call = llm_tool_call_raw["arguments"]
            if not isinstance(llm_tool_call, dict):
                print(llm_tool_call)
                raise ValueError("Tool call is not a valid dictionary")

            llm_tool_name = llm_tool_call_raw["name"]
        except KeyError:
            model_answer.score_raw_data[self.type] = "Correct JSON format but missing keys."
            return 0.0

        if llm_tool_name != api_call.api_description.name:
            model_answer.score_raw_data[self.type] = f"wrong_tool_name - LLM used the wrong tool name: {llm_tool_name}"
            return 0.0

        for param_name, param_type in api_call.api_description.parameters.items():
            # expected parameter value initially generated
            expected_param_value = api_call.parameters.get(param_name, None)

            if param_name == api_call.api_description.knowledge_parameter:
                valid, reason = self.match_knowledge_parameter(api_call, llm_tool_call)
                if not valid:
                    model_answer.score_raw_data[self.type] = reason
                    return 0.0
                llm_tool_call.pop(param_name)
                continue

            if expected_param_value is None:
                if perturbation_type == "omission" and param_name in api_call.original_parameters:
                    model_answer.score_raw_data[self.type] = f"Hallucinated parameter - LLM hallucinated a parameter not present in the user request"
                    return 0.0
                if self.debug:
                    print(param_name)
                    print(model_answer.answer)
                    print(api_call.parameters)
                    print(api_call.original_parameters)
                    print(api_call.perturbation_type)
                    print("API call not properly generated, set sample to error.")
                model_answer.iserror = True
                raise ValueError("API call not properly generated")

            # parameter missing from tool call
            if param_name not in llm_tool_call:
                model_answer.score_raw_data[self.type] = (
                    f"missing_parameter - Missing parameter in llm tool call: {param_name}"
                )
                return 0.0

            # model found out that some parameter is missing but still perform the tool call
            elif any(
                missing_str in str(llm_tool_call[param_name]).lower()
                for missing_str in ["missing", "missing parameter"]
            ):
                model_answer.score_raw_data[self.type] = f"missing_but_call - Model figured parameters are missing but still perform the tool call"
                return 0.0

            # parameter value mismatch, call an LLM to check if the values still agree
            elif not robust_string_match(str(llm_tool_call[param_name]), str(expected_param_value)):
                match, message = self.match_parameter_values(
                    api_call, llm_tool_call, param_name, expected_param_value
                )
                if not match:
                    model_answer.score_raw_data[self.type] = message
                    return 0.0

            llm_tool_call.pop(param_name)
        if len(llm_tool_call) == 0:
            model_answer.score_raw_data[self.type] = "correct_tool_call - LLM used the correct tool call"
            return 1.0

        # if there are extra parameters not matched in the tool call, these are extra parameters hallucinated by the model
        model_answer.score_raw_data[self.type] = (
            f"extra_parameters - Extra parameters in tool call: {llm_tool_call}"
        )
        return 0.0
