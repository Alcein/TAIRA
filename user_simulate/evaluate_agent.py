import json
import logging

import numpy as np
import pandas as pd
import yaml
from numpy import average
from pure_eval import Evaluator

from utils.task import extract_braces_content
from agents.agent import Agent


def calculate_mrr(ranked_items):
    """
    计算 Mean Reciprocal Rank (MRR)

    参数:
    ranked_items (list): 物品的排名列表，其中相关物品的值为相关性评分

    返回:
    float: MRR 值
    """
    # 找到第一个相关物品的排名
    reciprocal_ranks = []
    for rank, score in enumerate(ranked_items, start=1):
        if score > 0:  # 相关性评分大于 0 表示相关
            reciprocal_ranks.append(rank / rank)
            break

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_ndcg(ranked_items, p=10):
    """
    计算 Normalized Discounted Cumulative Gain (NDCG)

    参数:
    ranked_items (list): 物品的相关性评分列表
    p (int): 考虑的结果数量

    返回:
    float: NDCG 值
    """
    # 计算 DCG
    dcg = 0.0
    for i in range(min(p, len(ranked_items))):
        rel_i = ranked_items[i]
        dcg += rel_i / np.log2(i + 2)  # +2 因为索引从 0 开始，log2(1) = 0

    # 计算 IDCG
    ideal_relevance_scores = sorted(ranked_items, reverse=True)
    idcg = 0.0
    for i in range(min(p, len(ideal_relevance_scores))):
        rel_i = ideal_relevance_scores[i]
        idcg += rel_i / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


class EvaluateAgent(Agent):
    def __init__(self, memory, logger, config):
        super().__init__("EvaluateAgent", memory)
        self.logger = logger or logging.getLogger(__name__)
        self.config = config
        self.domain = config['DOMAIN']
        self.method = config['METHOD']
        self.domain_path = "data/" + self.domain

    def evaluate_one_recommend(self, query, answer, target_product, targets, preference):
        history = self.memory.get_history()
        rec_desc = answer["recommendation target"]

        product_info_list = []
        ids = [item['id'] for item in answer['items']]
        product_title = ''
        product_description = ''
        df = pd.read_csv(self.domain_path + '/metadata.csv')
        df['id'] = df['id'].astype(str)
        product_rows = df[df['id'].isin(ids)]
        for _, row in product_rows.iterrows():
            product_title = row['title']
            product_description = row['category']

            if self.domain == "amazon_clothing" or self.domain == "amazon_music":
                product_info = f"product: {product_title} description: {product_description}"
            elif self.domain == "amazon_beauty":
                product_info = f"product: {product_title} description: {product_description}"
            else:
                print('error')
                exit(0)
            product_info_list.append(product_info)
        # 用 '\n-----\n' 分隔拼接信息
        result_str = '\n-------------------\n'.join(product_info_list)

        sys_prompt = ("You are a user who is asking the conversational recommendation system for a target product. "
                      "You need a list of products that can truly meet your needs."
                      # f"Here is the previous task history:\n{history_str}\n"
                      )
        prompt = (
            f"Your requirement query is: \"{query}\"\n"
            "This query may require the system to recommend one or more products. "
            f"But now, the recommendation list you need to evaluate is only for this one recommend target: '{rec_desc}', "
            f"which is considered able to satisfy one of your targets. "
            "Imagine you are in this real-life situation and carefully understand your needs. "
            "Consider only the requirements you mention, don't add requirements that aren't mentioned out of thin air!"
            f"You will see a recommendation list of 10 items, all of which point to the same recommendation target, "
            f"which is {rec_desc} that has been judged above. "
            f"You should first determine whether each item in the recommendation list belongs to {rec_desc}."
            "Then determine whether this item can meet the needs of your query from some perspective."
            f"As long as it do, this is enough to prove that it meet your recommendation requirements."
            f"The recommend list for the recommend target given by the recommendation system is: \"{result_str}\" "
            
            f"In fact, there is such a sample product that can meet part of your requirement: \"{target_product}\". \n"
            "However, this is just an example, just because it can fulfil your requirements doesn't mean that "
            "its features are your requirements, you should still consider it based on your origin requirement statements.\n"

            "")

        json_format = """
            {
              "relevance_scores": [score1, score2, ... ,score10]
            }
        """
        prompt += (
            "Output a list of 10 ratings to express your judgment. "
            "The order in the rating list should correspond to the order of the items in the recommendation list. "
            "If it meets the requirements, it will correspond to 1 point, if it does not meet the requirements, "
            "it will correspond to 0 points. In particular, if it is exactly the same as the sample product, it will be given 2 points."
            "You can first output your reason, and then "
            f"output the final rating list in an json format:'{json_format}', score is a pure number."
        )
        if preference:
            prompt += (
                f"Additionally, your preference can be described as: \"{preference}\"\n"
                "You also need to decide whether each product meets your preferences in some way."
                "Then, among the products that are scored 1, change the score of those that do not meet your preferences to 0.5."
            )
        if self.domain == "amazon_music":
            prompt += ("When it comes to music recommendations, the satisfaction of many pieces of music may not "
                       "be reflected literally, so you don't have to be too strict. "
                       "In the case that the music does not conflict with the needs, you can consider giving 0 points less."
                       # "As long as the product is not completely unrelated to the demand, "
                       # "even if the description of the product is very broad, it can be considered to meet the demand."
                       # "And the judgment of user preferences can also be more loose."
                       )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        # response = get_completion(messages, llm='gpt-4o')
        response = self.call_gpt(messages)
        return response.strip()

    def evaluate_valid(self, query, desc_str, targets):
        sys_prompt = (
            "You are a user who is asking the conversational recommendation system for one or more product lists. "
            "Now you need to determine whether the topic of each recommended list given by the recommendation system meets your needs."
        )
        json_format = """
                    {
                      "valid_tags": [valid1,valid2,... ]
                    }
                """
        prompt = (
            f"The product targets provided by the recommendation system are:{desc_str}. Each target is separater by a ','. "
            f"Your query is: \"{query}\"\n"
            "This query may require the system to recommend one or more product lists. "
        )

        if targets != '':
            prompt += f"The recommended products should be of the following type targets: {targets}. \n"
        else:
            prompt += "These lists should meet your needs together. They are a whole and none of them can be missing. "
        prompt += (
            "Now, you need to judge whether each of these goals meets your requirements. "
            "If there is a target that does not meet your requirements, or is redundant or repeated, "
            "you should mark it as 0, otherwise mark it as 1."
            f"output the final rating list in an json format:'{json_format}', each valid is 0 or 1. "
            f"The number and order of tags is consistent with the type targets."
            f"Since product targets are non-empty, you must give at least one judgment."
            "You can't be strict. "
            "In the case that the product does not conflict with the needs, you can consider giving 0 points less."
        )
        if self.domain == "amazon_music":
            prompt += ("When it comes to music recommendations, the satisfaction of many pieces of music may not "
                       "be reflected literally, so you don't have to be too strict. "
                       "In the case that the music does not conflict with the needs, you can consider giving 0 points less.")
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        # print('prompt:', prompt)
        # response = get_completion(messages, llm='gpt-4o')
        response = self.call_gpt(messages)
        return response.strip()

    def evaluate(self, query, answer, target_product, targets, target_count, preference=None):
        fail_flag = False
        rec_number = len(answer["recommendations"])
        rec_descs = []
        for item in answer["recommendations"]:
            rec_desc = item["recommendation"]
            rec_descs.append(rec_desc)
        desc_str = ', '.join(rec_descs)
        if desc_str == '':
            valids = [0]*rec_number
        else:
            valids_str = extract_braces_content(self.evaluate_valid(query, desc_str, targets))
            # print(valids_str)
            valid_json = json.loads(valids_str)
            valids = valid_json["valid_tags"]
        # valids = [1] * rec_number
        self.logger.debug("valid: " + str(valids))
        if average(valids) <= 0.5:
            fail_flag = True
        if target_count == -1:
            if rec_number <= 1:
                fail_flag = True
            target_count = 2  # 失败的判定


        item_count = max(target_count, rec_number)
        hit = 0
        mrr = 0
        ndcg = 0
        for index, item in enumerate(answer["recommendations"]):
            if valids[index] == 0:
                hit += 0
                mrr += 0
                ndcg += 0
            else:
                rec_json = {"recommendation target": item["recommendation"], "items": item["items"]}
                result = self.evaluate_one_recommend(query, rec_json, target_product, targets, preference)
                self.logger.debug("evaluate of " + item["recommendation"] + ": " + result)
                result_str = extract_braces_content(result)
                result_json = json.loads(result_str)
                scores = result_json["relevance_scores"]
                hit += sum(scores)
                mrr += calculate_mrr(scores)
                ndcg += calculate_ndcg(scores)
        a_mrr = mrr / item_count
        a_ndcg = ndcg / item_count
        self.logger.debug("hit: " + str(hit))
        self.logger.debug("item_count: " + str(item_count))
        hit_rate = hit / item_count / 10
        return hit_rate, a_mrr, a_ndcg, fail_flag

        # recommend_list = [{"recommendation target": item["recommendation"], "items": item["items"]} for item in answer["recommendations"]]

    def execute_task(self, task):
        pass  # InteractorAgent does not execute tasks directly




