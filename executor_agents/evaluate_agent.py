import json
import logging

import numpy as np
import pandas as pd
import yaml

from utils.task import extract_braces_content
from .agent import Agent


def calculate_mrr(ranked_items):
    reciprocal_ranks = []
    for rank, score in enumerate(ranked_items, start=1):
        if score > 0:
            reciprocal_ranks.append(1 / rank)
            break

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_ndcg(ranked_items, p=10):
    dcg = 0.0
    for i in range(min(p, len(ranked_items))):
        rel_i = ranked_items[i]
        dcg += rel_i / np.log2(i + 2)
    ideal_relevance_scores = sorted(ranked_items, reverse=True)
    idcg = 0.0
    for i in range(min(p, len(ideal_relevance_scores))):
        rel_i = ideal_relevance_scores[i]
        idcg += rel_i / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


class EvaluateAgent(Agent):
    def __init__(self, memory, logger):
        super().__init__("EvaluateAgent", memory)
        self.logger = logger or logging.getLogger(__name__)
        with open('system_config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.domain = config['DOMAIN']
        self.method = config['METHOD']
        self.domain_path = "data/" + self.domain

    def evaluate_one_recommend(self, query, answer, target_product, complements, targets):
        history = self.memory.get_history()
        rec_desc = answer["recommendation target"]

        product_info_list = []
        ids = [item['id'] for item in answer['items']]
        product_title = ''
        product_description = ''
        if self.domain == "kdd22":
            df = pd.read_parquet(self.domain_path + '/shopping_queries_dataset_products.parquet')
            product_rows = df[df['product_id'].isin(ids)]

            for _, row in product_rows.iterrows():
                product_title = row['product_title']
                product_description = row['product_description']

                if len(product_description) > 100:
                    product_description = product_description[:100] + "..."
                product_info = f"product: {product_title} description: {product_description}"

                product_info_list.append(product_info)

        elif self.domain == "amazon_clothing":
            df = pd.read_parquet(self.domain_path + '/metadata.parquet')
            df['id'] = df['id'].astype(str)
            product_rows = df[df['id'].isin(ids)]

            for _, row in product_rows.iterrows():
                product_title = row['title']
                product_description = row['categories']
                product_info = f"product: {product_title} description: {product_description}"
                # print('product:', product_info)

                product_info_list.append(product_info)
        elif self.domain == "amazon_beauty":
            df = pd.read_parquet(self.domain_path + '/metadata.parquet')
            df['id'] = df['asin'].astype(str)
            product_rows = df[df['id'].isin(ids)]

            for _, row in product_rows.iterrows():
                product_title = row['title']
                product_description = row['categories'] + row['description']
                product_info = f"product: {product_title} description: {product_description}"
                product_info_list.append(product_info)

        result_str = '\n-----\n'.join(product_info_list)

        history_str = history
        sys_prompt = ("You are a user who is asking the conversational recommendation system for a certain need. "
                      "You need a product that can truly meet your needs."
                      )
        prompt = (
            f"Your complete requirements are: \"{query}\"\n"
            "This query may require the system to recommend one or more products. Some of them are main requirement that is explicitly specified; some are not, "
            "so the system needs to decide what kind of products to recommend. "
            "You need to first determine which situation your requirement belongs to, and whether there are other product requirements besides the main requirement. "
            "If so, please remember that the demand for each product is independent, "
            "which means that your judgment standard for whether other products meet the needs is not whether it is also another main requirement product. "
            f"In fact, there is such a sample product that can meet part of your requirement: \"{target_product}\". \n"
        )
        if complements != '':
            prompt += f"In addition, the following goods are complementary to the main demand: {complements}\n" \
                      "The types of goods involved here can all be matched with the target goods."
        prompt += (
            "However, these are just an example, just because it can fulfil your requirements doesn't mean that "
            "it's features are your requirements, you should still consider it based on your origin requirement statements.\n"
            "Imagine you are in this real-life situation and carefully understand your needs. "
            "Consider only the requirements you mention, don't add requirements that aren't mentioned out of thin air!"
            f"Now, the recommendation list you need to evaluate is only for this one recommend target: '{rec_desc}'. "
            "First, you should decide whether this recommended category meets your needs. "
        )
        if targets != '':
            prompt += f"The recommended products should be of the following types: {targets}. \n"
        prompt += (
            "This is just a recommendation for part of your recommendation needs. "
            "You need to judge whether it meets your requirements. "
            "Please remember again that if it is not the main requirement in your query, it does not mean that it does not meet the recommendation requirements"
            "As long as this recommended category meets part of your need (for example, your need is fishing, then any fishing tool is OK), "
            "it is considered 'yes' and you don't need to consider your main requirement. "
            "If it doesn't meet (It doesn't meet your requirements at all), you should output a point of 0."

            f"If yes, next you will see a recommendation list of 10 items, all of which point to the same recommendation target, "
            f"which is {rec_desc} that has been judged above. If {rec_desc} happens to be the main requirement product, "
            f"then you need to judge whether each product belongs to this type of product. "
            f"But, if {rec_desc} is not the main requirement, "
            f"then you should not judge whether these products belongs to the main requirement product, "

            f"because they are another type of product. Instead, you need to make judgement **only** based on whether a product belongs to {rec_desc} itself."
            f"And if {rec_desc} can meet your general requirement. "

            f"As long as it is, this is enough to prove that it meet your recommendation requirements."
            f"The recommend list for the recommend target given by the recommendation system is: \"{result_str}\" "
            "For some requirements, you don’t need to be too strict. For example, when the requirement is 'without something', "
            "**as long as the item description does not explicitly mention the existence of this thing, it can be assumed that it is without such thing!!!**"
            "Your judgment criteria should be very loose. As long as the product is related to the query, you can consider it to meet the requirements."
            )
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
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        response = self.call_gpt(messages)
        return response.strip()

    def evaluate(self, query, answer, target_product, complements, targets, target_count):
        fail_flag = False
        rec_number = len(answer["recommendations"])
        if target_count == -1:
            if rec_number <= 1:
                fail_flag = True
            target_count = 2

        item_count = max(target_count, rec_number)
        hit = 0
        mrr = 0
        ndcg = 0
        for item in answer["recommendations"]:
            rec_json = {"recommendation target": item["recommendation"], "items": item["items"]}
            result = self.evaluate_one_recommend(query, rec_json, target_product, complements, targets)
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


    def execute_task(self, task):
        pass
