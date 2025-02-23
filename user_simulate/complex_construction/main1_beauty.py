import json
import random
import re
import pandas as pd

from complex_query.main2 import extract_braces_content
from searcher import SearcherAgent
from analyzer_beauty import AnalyzeAgent

random.seed(43)


def main():
    # 读取文件
    input_file = '../dataset/beauty/query_beauty.csv'  # 修改为读取 query_data1.csv
    output_file = '../dataset/beauty/query_beauty1.csv'  # 输出到新的 CSV 文件

    # 读取文件
    df = pd.read_csv(input_file)
    # df = df.sample(n=10, random_state=2)

    # 初始化Agents
    # searcher_agent = SearcherAgent()
    analyzer_agent = AnalyzeAgent()

    # 初始化 classification 和 new_query 列
    df['new_query'] = ''

    # 处理每一行数据
    for index, row in df.iterrows():
        try:
            print("正在处理行", index)
            title = row['title']
            category = row['category']
            categories_list = category.split(' | ')
            cat_number = len(categories_list)
            item_info = analyzer_agent.extract_item(title, category)  # 物品概括
            # purpose = searcher_agent.search_knowledge(category, knowledge_file)  # 用途

            # 直接使用已有的 query 列
            query = row['query']  # 使用已有的 query
            new_query = query  # 初始化 new_query
            number = int(row['classification'])
            if number != -1:
                if cat_number >= 5:
                    number = 4  # 如果分类数大于等于10，直接设置 number
                else:
                    print(query)
                    query = analyzer_agent.examine_query(query)
                    print(query)
                    if cat_number >= 3:
                        num = random.randint(0, 3)
                        if num != 0:
                            classification = analyzer_agent.classify_query(query, item_info, num)
                            number = re.findall(r'\{(\d+)\}', classification)[-1]
                            new_query = analyzer_agent.rewrite_query(query, item_info, number, df)
                            if new_query is None or new_query == '':
                                new_query = analyzer_agent.rewrite_query(query, item_info, number, df, model='gpt-4o')
                            if new_query is None or new_query == '':
                                df = df.drop(index)
                                continue
                    else:
                        number = 0  # 分类数少于5时，设置 number 为 0
            else:
                print(query)
                new_query = analyzer_agent.examine_item(query)
                print(new_query)

            count_result = analyzer_agent.count_rec_number(new_query)
            count_json = json.loads(extract_braces_content(count_result))  # Use a safe JSON parser
            count = count_json["product_count"]


            if count != -1:
                product_list = '|'.join(count_json["product_list"])
            else:
                product_list = ''
            # 更新 classification 和 new_query 列
            df.at[index, 'classification'] = number
            df.at[index, 'target_count'] = count
            df.at[index, 'targets'] = product_list
            df.at[index, 'new_query'] = new_query
            df.loc[df['classification'].isin(['1', '3']), 'target_count'] = -1
            df.loc[df['classification'].isin(['1', '3']), 'targets'] = None
        except Exception as e:
            df.drop(index, inplace=True)
            print(f"处理行 {index} 时发生错误: {e}")  # 输出错误信息

    # 保存结果到新的文件
    df.to_csv(output_file, index=False)
    print(f"Queries saved to {output_file}")


if __name__ == "__main__":
    main()
