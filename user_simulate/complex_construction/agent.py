# agents/agent.py

from abc import ABC, abstractmethod
from utils.task import get_completion


class Agent(ABC):
    def __init__(self, name):
        self.name = name
        # self.memory = memory

    @abstractmethod
    def execute_task(self, task):
        pass

    def call_gpt(self, messages, llm=None, json_format=None):
        return get_completion(messages, json_format, llm)
