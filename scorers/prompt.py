from lmeval.enums import TaskType
from lmeval.prompts.prompt import Prompt
from lmeval.task import Task
from lmeval.question import Question, GroupedQuestion

class CompletionPrompt(Prompt):
    def __init__(
        self,
        template: str = "",
        name: str = "Completion Prompt",
        description: str = "Ask the model to complete the given conversation",
        task_type=TaskType.completion,
        url: str = "",
        version: str = "1.0",
    ):
        super().__init__(
            name=name,
            description=description,
            task_type=task_type,
            template=template,
            url=url,
            version=version,
        )

    def render(self, question: Question, task: Task) -> str:  # noqa: F821
        "Render prompt for a given question and task"

        if task.type != self.task_type:
            raise ValueError(
                f"Task type {task.type} does not match prompt task type {self.task_type}"
            )

        return question.messages


class GroupedCompletionPrompt(Prompt):
    def __init__(
        self,
        template: str = "",
        name: str = "Grouped Completion Prompt",
        description: str = "Ask the model to complete the given conversation",
        task_type=TaskType.grouped_completion,
        url: str = "",
        version: str = "1.0",
    ):
        super().__init__(
            name=name,
            description=description,
            task_type=task_type,
            template=template,
            url=url,
            version=version,
        )

    def render(self, question: GroupedQuestion, task: Task) -> str:  # noqa: F821
        "Render prompt for a given question and task"

        if task.type != self.task_type:
            raise ValueError(
                f"Task type {task.type} does not match prompt task type {self.task_type}"
            )

        return question.messages
