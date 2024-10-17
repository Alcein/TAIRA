from .agent import Agent


class InteractorAgent(Agent):
    def __init__(self, memory):
        super().__init__("InteractorAgent", memory)

    def generate_response(self, instruction):
        history = self.memory.get_history()
        history_str = history
        sys_prompt = ("You are a response agent of a conversational recommendation system."
                      "You are good at analyzing provided information and generate recommendation response."
                      f"Here is the previous task history:\n{history_str}\n"
                      )
        json_format = """
        {
            "recommendations": [
                {
                    "recommendation": "...",
                    "items": [
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."}
                    ]
                },
                {
                    "recommendation": "...",
                    "items": [
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."}
                    ]
                },
                {
                    "recommendation": "...",
                    "items": [
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."},
                        {"id": "...", "title": "..."}
                    ]
                },
                ......
            ]
        }
        """
        prompt = (
            f"Based on the task history, and the instruction from manager:{instruction}"
            "If you've got enough recommend list, "
            "generate a response with one or ore lists, each list containing 10 recommended items (id and title). "
            "You need to correctly understand the intent in the **complete** task history and include a list of **all** "
            "the recommendations needed in the final response. Especially when there are multiple plans for the task execution, "
            "**don't just recommend items retrieved in the last plan!**"
            f"Output the lists using the following JSON format:\n{json_format}\n"
            "In the 'recommendation', you should use no more than 5 words to describe the basic type of product you are recommending, "
            "especially the product category. Then the 'items' is a list of recommendations for this target. "
            "In item information, you must keep as many keywords as possible in the input words when searching for these items. "
            "You cannot remove these keywords because they will be used to evaluate the quality of recommendations."
            "You must output 10 items for each list."
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        response = self.call_gpt(messages)

        return response.strip()

    def execute_task(self, task):
        pass
