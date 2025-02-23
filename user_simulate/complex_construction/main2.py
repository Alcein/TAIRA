import json
import re
import pandas as pd
from searcher import SearcherAgent
from analyzer import AnalyzeAgent


def extract_braces_content(s):
    # 使用正则表达式匹配最前面的{和最后面的}之间的所有内容
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def main():
    # 读取文件
    input_file = '../KDD22/kdd_query0.csv'  # 修改为读取 query_data1.csv
    output_file = '../KDD22/kdd_query1.csv'  # 输出到新的 CSV 文件
    # knowledge_file = 'E:/Code-Agent/multi_agent/knowledge/knowledge0.csv'

    # 读取文件
    df = pd.read_csv(input_file)
    # df = df.iloc[82:85]

    # 初始化Agents
    # searcher_agent = SearcherAgent()
    analyzer_agent = AnalyzeAgent()

    # 初始化 classification 和 new_query 列
    df['classification'] = ''
    df['new_query'] = ''
    df['target_count'] = ''
    df['targets'] = ''
    df['complements'] = ''
    products_df = pd.read_parquet('E:/Code-Agent/multi_agent/data/kdd22/shopping_queries_dataset_products.parquet')

    # 处理每一行数据
    for index, row in df.iterrows():
        print("正在处理行", index)
        try:
            # 检查 row['c_product_ids'] 是否为空或 None
            if not row['c_product_ids'] or pd.isna(row['c_product_ids']):
                # 如果 c_product_ids 为空或 NaN，跳过处理或设定为空列表
                c_product_ids = []
            else:
                # 如果不为空，按照 '|' 分割 c_product_ids
                c_product_ids = str(row['c_product_ids']).split('|')

            # 初始化空列表用于存储 c_product_info
            c_product_info = []
            concatenated_texts = ''
            # 遍历每个 c_product_id，在 products_df 中查找对应的 product_title 和 product_description
            for c_product_id in c_product_ids:
                # 跳过空的 c_product_id
                if c_product_id == '':
                    continue

                # 查找产品信息
                product_row = products_df[products_df['product_id'] == c_product_id]

                # 如果找到了对应的产品
                if not product_row.empty:
                    # 取出 product_title 和 product_description
                    product_title = product_row.iloc[0]['product_title']
                    product_description = product_row.iloc[0]['product_description']

                    # 拼接 title 和 description 并加入列表
                    product_info = f"item: {product_title} description: {product_description}"
                    c_product_info.append(product_info)
                    concatenated_texts = '\n-----\n'.join(c_product_info)

            product_info = row['product_info']
            # 直接使用已有的 query 列
            query = row['query']  # 使用已有的 query
            complete_query = analyzer_agent.complement_query(query)  # 物品概括

            # 初始化 new_query
            classification = analyzer_agent.classify_query_product(complete_query, product_info)
            number = re.findall(r'\{(\d+)\}', classification)[-1]

            new_query = analyzer_agent.rewrite_query_product(query, number, concatenated_texts)
            count_result = analyzer_agent.count_rec_number(new_query)
            count_json = json.loads(extract_braces_content(count_result))  # Use a safe JSON parser
            count = count_json["product_count"]

            if count != -1:
                product_list = '|'.join(count_json["product_list"])
            else:
                product_list = ''

            # 正常处理后填入数据
            df.at[index, 'classification'] = number
            df.at[index, 'new_query'] = new_query
            df.at[index, 'target_count'] = count
            df.at[index, 'targets'] = product_list
            df.at[index, 'complements'] = concatenated_texts

        except Exception as e:
            print(f"行 {index} 处理时出错: {e}")
            # 发生异常时填入 'error'
            df.at[index, 'classification'] = 'error'
            df.at[index, 'new_query'] = 'error'
            df.at[index, 'target_count'] = 'error'
            df.at[index, 'targets'] = 'error'
            df.at[index, 'complements'] = 'error'

    # 保存处理结果
    df.to_csv(output_file, index=False)
    print(f"Queries saved to {output_file}")


if __name__ == "__main__":
    main()
