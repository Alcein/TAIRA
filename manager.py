import logging
import re

import torch
import yaml

from executor_agents.evaluate_agent import EvaluateAgent
from utils.task import get_completion
from utils.memory import Memory
from utils.Prompts import ONE_PLAN_PROMPT, BoT_AGENTS_INSTRUCTION, AGENTS_INSTRUCTION
from utils.thought_template import thought_templates
import json


def extract_braces_content(s):
    s = s.replace("\\'", "'")
    # 使用正则表达式匹配最前面的{和最后面的}之间的所有内容
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


class Manager:
    def __init__(self, memory, user_input, target_product, complements, targets, target_count, logger=None):
        self.turn = 0
        self.agents = {}
        self.memory = memory
        self.user_input = user_input
        self.target_product = target_product
        self.templates = thought_templates
        self.complements = complements
        self.targets = targets
        self.target_count = target_count
        self.logger = logger or logging.getLogger(__name__)
        with open('system_config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.method = self.config['METHOD']

    def register_agent(self, agent):
        self.agents[agent.name] = agent

    def plan_task(self, template):
        # Here, we generate the task plan as a JSON structure
        history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        sys_prompt = (
            "You are a manager agent of a conversational recommendation system."
            "You are good at analyzing user inquiry intent and planning tasks."
            "In addition, you are good at transferring the high-level thinking processes "
            "of previous successful experiences to current problems."
            "Here are the available executor_agents and their functionalities:\n"
            f"{BoT_AGENTS_INSTRUCTION}"
            # f"{AGENTS_INSTRUCTION}"
        )
        prompt = (
            f"The user's input is: \"{self.user_input}\". "
            "Based on the user's input, create a task plan in JSON format with sub-tasks."
            "\nThe output should be JSON format as follows:\n"
            "{ \n"
            f"  \"user_input\": \"{self.user_input}\" ,\n"
            "  \"main_task\": \"...\" ,\n"
            "  \"sub_tasks\": { \n"
            "    \"task_1\": {\"content\": \"...\" , \"agent\": \"...\"}, \n"
            "    \"task_2\": {\"content\": \"...\" , \"agent\": \"...\"}, \n"
            "       ......"
            "  } \n"
            "}"
            "'Content' is what the agent should do. And 'agent' specifying the agent to execute each sub-task."
            "Remember: PlannerAgent and InteractorAgent **must** be the last sub-task in the plan. "
            "No sub-tasks are allowed after a task assigned to either PlannerAgent or InteractorAgent."
            "You can only use PlannerAgent or InteractorAgent once, and it must be in the final sub-task. "
            "There should be no sub-tasks after that."
            "If you think the current task can be completed with a single plan, choose an InteractorAgent, "
            "otherwise, choose a PlannerAgent to update the plan after getting enough information."
        )
        if self.method == "COT":
            prompt += (
                "You must first output the Chain of Thoughts ( COT ) . "
                "In the COT , you need to explain how you break down the main task into sub - tasks "
                "and justify why each subtask can be completed by a tool ."
            )
        elif self.method == "TAIRA":
            prompt += (
                "You need to follow the following thinking template to complete the task. "
                "This template is a high-level thinking process summarized from the successful experience of similar tasks:"
                f"{template}"
            )

        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        if self.config['MODEL'] == 'qwen-plus-2024-09-19':
            response = get_completion(messages, 'qwen-max')
        else:
            response = get_completion(messages)
        # response = get_completion(messages)
        task_plan = response.strip()
        return task_plan

    def re_plan_task(self, template, task):
        # Here, we generate the task plan as a JSON structure
        history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        sys_prompt = (
            "You are a manager agent of a conversational recommendation system."
            "You are good at analyzing user inquiry intent and planning tasks."
            "In addition, you are good at transferring the high-level thinking processes "
            "of previous successful experiences to current problems."
            "Here are the available executor_agents and their functionalities:\n"
            f"{BoT_AGENTS_INSTRUCTION}"
            "The following is the history of tasks executed so far: \n"
            f"{history}"
            f"First, you should decide your plan goal based on the mission history."
            f"Then, you need to generate a follow-up to the previous plan to continue completing the task."
        )
        prompt = (
            f"Your plan goals are: \"{task}\". "
            "Based on the user's input, previous plan and the execution history, "
            "generate a follow-up to the previous plan with sub-tasks in JSON format."
            "\nThe output should be JSON format as follows:\n"
            "{ \n"
            "  \"sub_tasks\": { \n"
            "    \"task_1\": {\"content\": \"...\" , \"agent\": \"...\"}, \n"
            "    \"task_2\": {\"content\": \"...\" , \"agent\": \"...\"}, \n"
            "       ......"
            "  } \n"
            "}"
            "'Content' is what the agent should do. And 'agent' specifying the agent to execute each sub-task."
            "Remember: PlannerAgent and InteractorAgent **must** be the last sub-task in the plan. "
            "No sub-tasks are allowed after a task assigned to either PlannerAgent or InteractorAgent."
            "You can only use PlannerAgent or InteractorAgent once, and it must be in the final sub-task. "
            "There should be no sub-tasks after that."
            "If you think the current task can be completed with a single plan, choose an InteractorAgent, "
            "otherwise, choose a PlannerAgent to update the plan after getting enough information."
        )
        if self.method == "COT":
            prompt += (
                "You must first output the Chain of Thoughts ( COT ) . "
                "In the COT , you need to explain how you break down the main task into sub - tasks "
                "and justify why each subtask can be completed by a tool ."
            )
        elif self.method == "TAIRA":
            prompt += (
                "You need to continue to follow the following thinking template to complete the task. "
                "This template is a high-level thinking process summarized from the successful experience of similar tasks:"
                f"{template}"
            )
        # print(sys_prompt, prompt)

        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        if self.config['MODEL'] == 'qwen-plus-2024-09-19':
            response = get_completion(messages, 'qwen-max')
        else:
            response = get_completion(messages)
        # response = get_completion(messages)
        task_plan = response.strip()
        return task_plan

    def delegate_task(self):
        self.logger.debug("user query: " + self.user_input)
        evaluator = EvaluateAgent(self.memory, self.logger)
        self.memory.add_input(self.user_input)
        if self.method == "TAIRA":
            self.logger.debug("choosing template...")
            chosen_number = self.select_template()  # 替换为你实际想要的索引
            print(chosen_number)
            chosen_number = int(re.findall(r'\{(\d+)\}', chosen_number)[-1]) + 1
            # 提取对应的内容
            self.logger.debug("template_number:" + str(chosen_number))
            template_key = f"template_{chosen_number}"
            template = self.templates[template_key]
            solution_description = template['solution_description']
            thought_template = template['thought_template']

            # 拼接内容
            template = (
                # f"solution_description: {solution_description}\n"
                f"thought_template: {thought_template}"
            )

            # 打印或返回拼接后的内容
            # print(template)
            self.logger.debug('template: ' + template)
            # return template_key
            self.logger.debug('creating plan...')
        else:
            chosen_number = 0
            template = ''
        task_plan = self.plan_task(template)
        # print("Task Plan:", task_plan)
        plan = extract_braces_content(task_plan)
        interpreter_agent = self.agents.get("InterpreterAgent")
        interactor_agent = self.agents.get("InteractorAgent")
        task_plan_json = json.loads(plan)  # Use a safe JSON parser
        self.memory.add_plan(plan)
        self.logger.debug('plan: ' + plan)
        self.logger.debug('executing plan...')
        last_name = ''
        output = ''
        # print(task_plan)
        while last_name != "InteractorAgent":
            for sub_task_key in list(task_plan_json["sub_tasks"])[:-1]:
                sub_task = task_plan_json["sub_tasks"][sub_task_key]
                content = sub_task["content"]
                agent_name = sub_task["agent"]
                query = interpreter_agent.process_output(content, agent_name, output)
                agent = self.agents.get(agent_name)

                if agent:
                    if self.turn >= 10:
                        self.status = False
                        # print("fail")
                        self.logger.debug("fail")
                        return 0, 0, 0, True, 0
                    self.turn += 1
                    origin_output = agent.execute_task(query)
                    output = str(origin_output)
                    # print("output:", output)
                    self.memory.add_observation(agent_name, query, output)
                    self.logger.debug('agent: ' + agent_name)
                    self.logger.debug('query: ' + query)
                    self.logger.debug('output: ' + output)
                else:
                    print('no agent')
                    return 0, 0, 0, True, 0

            last_task_key = list(task_plan_json["sub_tasks"])[-1]
            last_task = task_plan_json["sub_tasks"][last_task_key]
            last_name = last_task["agent"]
            last_content = last_task["content"]
            if last_name == "InteractorAgent":
                query = interpreter_agent.process_output(last_content, last_name, output)
                self.logger.debug('recommendation command: ' + query)
                result = interactor_agent.generate_response(query)
                print(result)
                self.logger.debug('recommendation: ' + result)
                items = extract_braces_content(result)
                rec_json = json.loads(items)
                hit_rate, mrr, ndcg, fail_flag = evaluator.evaluate(self.user_input, rec_json, self.target_product,
                                                                    self.complements, self.targets, self.target_count)
                if hit_rate == 0:
                    fail_flag = True
                self.logger.debug('hit rate: ' + str(hit_rate))
                self.logger.debug("mrr: " + str(mrr))
                self.logger.debug("ndcg: " + str(ndcg))
                return hit_rate, mrr, ndcg, fail_flag, chosen_number
            elif last_name == "PlannerAgent":
                query = interpreter_agent.process_output(last_content, last_name, output)
                self.logger.debug('plan goal:  ' + query)
                new_task_plan = self.re_plan_task(template, query)
                # print("NEW Task Plan:", new_task_plan)
                new_plan = extract_braces_content(new_task_plan)
                old_json = task_plan_json

                task_plan_json = json.loads(new_plan)  # Use a safe JSON parser

                old_json["sub_tasks"].update(task_plan_json["sub_tasks"])
                new_plan = json.dumps(old_json, indent=2, ensure_ascii=False)
                self.memory.add_plan(new_plan)
                self.logger.debug('plan: ' + new_plan)

    # def analyze_query(self):
    #     history = self.memory.get_history()
    #     # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
    #     sys_prompt = (
    #         "As a highly professional and intelligent expert in information distillation, you excel at "
    #         "extracting essential information to solve problems from user queries in conversational recommendation system. "
    #         "You need to come up with a general form of meta-question that can format user queries and "
    #         "handle more input and output variations. "
    #     )
    #     prompt = (
    #         f"The user's input is: \"{self.user_input}\". "
    #         "Only output the meta-question."
    #     )
    #
    #     messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    #     response = get_completion(messages)
    #     meta_query = response.strip()
    #     return meta_query
    #
    # def compute_similarity(self, query, texts, tokenizer, model, batch_size=128, device='cuda'):
    #     all_scores = []
    #     model.to(device)
    #     for i in range(0, len(texts), batch_size):
    #         batch_texts = texts[i:i + batch_size]
    #         pairs = [[query, text] for text in batch_texts]
    #         with torch.no_grad():
    #             inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
    #             scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    #             all_scores.extend(scores.cpu().numpy())
    #     return all_scores
    #
    # def select_template(self, query, tokenizer, model, device='cuda'):
    #     # 获取所有模板的 task_description 和 solution_description 的组合
    #     template_texts = [
    #         template['task_description'] + "\n" + template['solution_description']
    #         for template in self.templates.values()
    #     ]
    #
    #     # 计算 query 与所有模板的相似度
    #     similarities = self.compute_similarity(query, template_texts, tokenizer, model, device=device)
    #
    #     # 找到相似度最高的模板
    #     best_index = similarities.index(max(similarities))
    #
    #     # 返回相似度最高的模板
    #     best_template_key = list(self.templates.keys())[best_index]
    #     best_template = self.templates[best_template_key]
    #
    #     return best_template_key, best_template

    def select_template(self):
        history = self.memory.get_history()
        template_texts = [
            f"template {index}: {template['task_description']}"  # 只保留 task_description，并加上序号
            for index, template in enumerate(self.templates.values())
        ]
        combined_template_text = "\n".join(template_texts)
        # print(combined_template_text)
        sys_prompt = (
            "You are an expert at finding the right experience from past successful experiences to solve current problems. "
            "Here are some descriptions of successful experiences you can learn from:"
            f"{combined_template_text}"
        )
        prompt = (
            f"The current user query is: \"{self.user_input}\". "
            "Please give the number of a template that you think is most similar to the current task. "
            "output the chosen number in an '{}', '{}' contains PURE number without any signs."
        )

        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        response = get_completion(messages)
        meta_query = response.strip()
        if '{' not in meta_query:
            meta_query = f'{{{meta_query}}}'
        return meta_query
