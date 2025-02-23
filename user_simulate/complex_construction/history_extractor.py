import json

import pandas as pd
from pydantic import BaseModel

from agent import Agent
from collections import Counter

from complex_query.utils.task import get_completion

domain = 'music'

class HistoryFormat(BaseModel):
    profile: str
    preference: str


class HistoryExtractor(Agent):
    def __init__(self):
        super().__init__("HistoryExtractor")
        self.domain = domain
        self.history_file = f'../dataset/{self.domain}/history_index_metadata.csv'


    def process_items(self, item_ids, n):
        # 解析输入的item_ids
        ids = item_ids.split('|')

        # 加载CSV文件
        history_df = pd.read_csv(self.history_file)

        # 用于存储所有标题和所有类别的关键词
        all_titles = []
        all_categories = []
        sum_price = 0
        count = 0

        # 遍历每个item_id
        for item_id in ids:
            # 获取该item_id对应的行
            # item_row = history_df[history_df['id'] == item_id]
            item_row = history_df[history_df['id'] == item_id]

            if not item_row.empty:
                # 获取标题并添加到all_titles
                title = item_row['title'].values[0]
                all_titles.append(title)

                # 获取类别并添加到all_categories
                categories = item_row['category'].values[0]
                # 清理类别中的空格并拆分成关键词
                category_keywords = [category.strip() for category in categories.split('|')]
                all_categories.extend(category_keywords)

                price = item_row['price'].values[0]
                if not pd.isnull(price):
                    try:
                        print(price)
                        sum_price += float(price)
                        count += 1
                    except Exception as e:
                        print(price)
                        print(e)


        # 拼接所有的title，按顺序用逗号分隔
        items = ', '.join(all_titles)

        # 统计类别关键词的频次
        category_counter = Counter(all_categories)

        # 获取前n个最常见的关键词
        top_n_keywords = [item[0] for item in category_counter.most_common(n)]

        # 拼接top n关键词，按频次从高到低，用逗号分隔
        preferences = ', '.join(top_n_keywords)
        print("sum:", sum_price)
        print("count:", count)
        if count:
            average_price = sum_price / count
        else:
            average_price = None

        return items, preferences, average_price

    def extract_history(self, items, preferences):  # 把history序列转化为profile
        sys_prompt = """
        You are an expert in user behavior analysis for recommendation systems,
        You are good at extracting user preferences and behavior characteristics from their interaction history.
        """
        json_str = (
            "{\n"
            "  \"profile\": ...,\n"
            "  \"preference\": ...\n"
            "}\n"
        )
        if self.domain == 'cloth':
            special = "the user's gender, body shape, significantly strong preferences and other basic information"
        if self.domain == 'music':
            special = "the user's gender, age group, significantly strong preferences and other basic information"
        if self.domain == 'beauty':
            special = "the user's gender, age group, significantly strong preferences and other basic information"
        prompt = f"""
        The products that users recently purchased and liked are: {items}
        The most frequent keywords in the products that users purchased and liked are: {preferences}. These are ordered from most frequent to least frequent.
        First, please give a simple description of the user, including but not limited to {special}. You cannot guess, but can only give a relatively certain summary.
        Second, summarize the user's unique preferences from the user's recent keywords. Please note that general keywords that describe the most basic types of products and the basic information of users that have been summarized should not be included.
        You should use the given structure for output. Each should be no more than 3 sentences. If there is less content that can be determined, do not output more than necessary.
        No need to output detail reason of the information.
        """
        # ：{json_str}
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        # response = self.call_gpt(messages, json_format=HistoryFormat)
        response = get_completion(messages, json_format=HistoryFormat)
        print(f"Chat Response: {response}")
        return response

    def execute_task(self, task):
        return


def main():
    # interactions_df = pd.read_csv('../dataset/cloth/user_interaction.csv').head(10)
    interactions_df = pd.read_csv(f'../dataset/{domain}/user_interaction.csv')
    agent = HistoryExtractor()
    for index, row in interactions_df.iterrows():
        interactions = row['interaction_string'].split('|')
        inter_history = '|'.join(interactions[1:])
        # print(inter_history)
        items, preferences, price = agent.process_items(inter_history, 5)
        # print(items)
        # print(preferences)
        try:
            response = json.loads(agent.extract_history(items, preferences))
        except Exception as e:
            print("错误", e)
            interactions_df.drop(index, inplace=True)
            continue
        profile = response['profile']
        preferences = response['preference']
        interactions_df.at[index, 'profile'] = profile
        interactions_df.at[index, 'preferences'] = preferences
        interactions_df.at[index, 'price'] = price
        # print(response)
        # print(profile)
        # print(preferences)
        # print(price)
    # interactions_df.to_csv('../dataset/cloth/user_profile.csv')
    interactions_df.reset_index(drop=True, inplace=True)
    interactions_df.to_csv(f'../dataset/{domain}/user_profile.csv')




if __name__ == "__main__":
    main()
