class Memory:
    def __init__(self):
        self.history = []

    def add_input(self, user_input):
        self.history.append({'user query': user_input})

    def add_plan(self, plan):
        self.history.append({'current plan': plan})

    def add_observation(self, agent, task, output):
        self.history.append({'agent': agent, 'selected task': task, 'result': output})

    def add_thought(self, thought):
        self.history.append(thought)

    def get_history(self):
        return self.history

    def remove_data(self):
        self.history = []
