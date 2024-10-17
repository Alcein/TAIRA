from .agent import Agent
from utils.Prompts import BoT_AGENTS_INSTRUCTION


class InterpreterAgent(Agent):
    def __init__(self, memory):
        super().__init__("InterpreterAgent", memory)

    def process_output(self, content, next_agent_name, output, template = ''):
        history = self.memory.get_history()
        history_str = history
        sys_prompt = (
            "You are a task planning agent of a conversational recommendation system."
            "You are good at analyzing user inquiry intent and planning tasks."
            f"{BoT_AGENTS_INSTRUCTION}"
            f"Here is the previous task history:\n{history_str}\n"
        )
        prompt = (
            f"The current task is \"{content}\""
            f"The next agent to complete this task is: \"{next_agent_name}\". "
            f"The previous task output is: \"{output}\". "
            "Based on these information, generate the query for the next agent to make sure it can complete the task and generate right output."
            "Remember only output the query!"
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        response = self.call_gpt(messages)
        return response.strip()

    def execute_task(self, task):
        pass  # InteractorAgent does not execute tasks directly
