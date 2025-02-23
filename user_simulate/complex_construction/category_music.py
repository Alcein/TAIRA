import time
import pandas as pd
from complex_query.searcher import SearcherAgent

# 读取CSV文件
df = pd.read_csv('../dataset/music/music.csv')
# df = df.head(5000)

# 提取categories列，并去掉NaN值
categories_column = df['category'].dropna()

# 初始化一个空列表，用于存储不重复的分类短语
unique_categories_list = []

# 遍历categories列，分割每个条目，并添加到列表中
for categories in categories_column:
    categories_list = categories.split(' | ')
    for category in categories_list:
        if category not in unique_categories_list:  # 检查是否已经存在
            unique_categories_list.append(category)  # 添加唯一元素

# 输出分类短语的总数
print(f"总共有 {len(unique_categories_list)} 个分类短语")

# # 输出不重复的分类短语列表
# print(unique_categories_list)
#
#
# searcher = SearcherAgent()
#
# data = []
#
# test_list1 = unique_categories_list
#
# for index, attribute in enumerate(test_list1, 1):
#     print("正在检索词", index)
#     try:
#         usage = searcher.execute_search_music(attribute)
#         usage = usage.replace('\n', ' ').replace('\r', ' ')  # 去掉换行符
#         data.append({'attribute': attribute, 'usage': usage})
#     except Exception as e:
#         print(f"检索属性 '{attribute}' 时发生错误: {e}")  # 输出错误信息
#     time.sleep(1)
#
# # 将数据转换为 DataFrame
# df = pd.DataFrame(data)
#
# # 保存 DataFrame 为 CSV 文件
# df.to_csv('../knowledge/music_knowledge0.csv', index=False)
