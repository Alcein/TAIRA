import random

import pandas as pd
from searcher import SearcherAgent
from analyzer import AnalyzeAgent
from history_extractor import HistoryExtractor

random.seed(42)


def main():
    # 读取文件
    input_file = '../dataset/cloth/merged_metadata.csv'
    output_file = '../dataset/cloth/query_data0.csv'
    user_interaction_file = '../dataset/cloth/user_profile.csv'
    knowledge_file = '../knowledge/knowledge0.csv'

    # 读取metadata文件和用户交互文件
    metadata_df = pd.read_csv(input_file)
    user_interaction_df = pd.read_csv(user_interaction_file)

    # 初始化Agents
    searcher_agent = SearcherAgent()
    analyzer_agent = AnalyzeAgent()

    # 存储最终结果的列表
    result_rows = []

    # 处理每一行的用户交互数据
    for _, user_row in user_interaction_df.iterrows():
        print(f"正在处理用户 {user_row['user_id']} 的交互数据")

        user_id = user_row['user_id']
        interaction_string = user_row['interaction_string']
        user_profile = user_row['profile']
        user_preference = user_row['preferences']

        # 获取最新的两个交互item的id
        interactions = interaction_string.split('|')

        # if len(interactions) > 6:
        #     item_ids = interactions[:3]
        # else:
        #     item_ids = interactions[:2]
        #
        # if len(item_ids) < 2:
        #     print(f"用户 {user_id} 的交互记录不足两个项目")
        #     continue
        item_id = interaction_string.split('|')[0]
        item_rows = metadata_df[metadata_df['id'] == item_id]
        if len(item_rows) == 0:
            continue
        item_row = item_rows.iloc[0]

        title = item_row['title']
        category = item_row['category']
        categories_list = category.split(' | ')
        cat_number = len(categories_list)
        query = ''

        item_info = analyzer_agent.extract_item(title, category)  # 物品概括
        if random.random() < 1 / 3:
            while query is None or query == '':
                query = analyzer_agent.item_query(item_info, cat_number, user_profile)
            number = -1
        else:
            purpose = searcher_agent.search_knowledge(category, knowledge_file)  # 用途
            while query is None or query == '':
                query = analyzer_agent.generate_query(item_info, purpose, cat_number, user_profile)
            number = 0
        remain_history = interactions[1:]

        # 创建item_row的副本并添加新列
        new_item_row = item_row.copy()

        new_item_row['user_id'] = user_id
        new_item_row['remaining_interaction_string'] = '|'.join(remain_history)
        new_item_row['query'] = query
        new_item_row['classification'] = number
        new_item_row['preferences'] = user_preference

        # 将更新后的新行加入结果列表
        result_rows.append(new_item_row)

        # except Exception as e:
        #     print(f"处理用户 {user_id} 的交互时发生错误: {e}")  # 输出错误信息

    # 将结果转为DataFrame
    result_df = pd.DataFrame(result_rows)

    # 保存结果到新的文件
    result_df.to_csv(output_file, index=False)
    print(f"Queries saved to {output_file}")


if __name__ == "__main__":
    main()
