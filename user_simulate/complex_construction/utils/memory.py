# utils/memory.py
class Memory:
    def __init__(self):
        self.history = []

    def add_input(self, user_input):
        self.history.append({'user': user_input})

    def add_task(self, task):
        self.history.append({'task': task})

    def add_observation(self, agent, task, output):
        self.history.append({'agent': agent, 'task': task, 'result': output})

    def get_history(self):
        return self.history
