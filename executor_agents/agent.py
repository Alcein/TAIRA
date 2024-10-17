from abc import ABC, abstractmethod

import yaml

from utils.task import get_completion


class Agent(ABC):
    def __init__(self, name, memory):
        with open('system_config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.model = self.config['MODEL']
        self.name = name
        self.memory = memory

    @abstractmethod
    def execute_task(self, task):
        pass

    def call_gpt(self, messages, llm=None):  # claude-3-5-sonnet-20240620 gpt-4o-2024-08-06 qwen-plus
        return get_completion(messages, llm=self.model)
