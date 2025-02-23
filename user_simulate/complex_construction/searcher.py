# agents/searcher.py
import time

from agent import Agent
import requests
from requests.exceptions import ConnectionError, HTTPError
from utils.task import get_completion
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SearcherAgent(Agent):
    def __init__(self):
        super().__init__("SearcherAgent")
        self.api_key = 'AIzaSyD_q4klu7UkPz-e3QdMmQGuFzXdH2Kj4tM'
        self.cse_id = '70832507ec0774bc2'
        self.max_retries = 2
        self.backoff_factor = 2

    def google_search(self, query, num_results=5):
        url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.cse_id}&q={query}"
        retries = 0

        while retries < self.max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()  # Ensure request was successful
                results = response.json().get('items', [])
                return self.extract_and_format_info(results[:num_results])
            except ConnectionError as e:
                print(f"Connection error: {e}, retrying ({retries + 1}/{self.max_retries})...")
            except HTTPError as e:
                if response.status_code == 429:  # Too Many Requests
                    print(f"Rate limit exceeded: {e}, retrying ({retries + 1}/{self.max_retries})...")
                else:
                    print(f"HTTP error: {e}")
                    return None
            time.sleep(self.backoff_factor ** retries)  # Exponential backoff
            retries += 1

        print("Max retries reached. Failed to retrieve search results.")
        return None

    def extract_and_format_info(self, results):
        useful_info = []
        for item in results:
            info = {
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', '')
            }

            # 提取pagemap中的其他内容
            pagemap = item.get('pagemap', {})
            for key, value in pagemap.items():
                if key not in ['cse_thumbnail', 'cse_image']:
                    info[key] = value

            useful_info.append(info)
        formatted_info = self.format_info(useful_info)
        return formatted_info

    def format_info(self, useful_info):
        formatted_info = []
        for info in useful_info:
            formatted_str = f"Title: {info.get('title', '')}\nLink: {info.get('link', '')}\nSnippet: {info.get('snippet', '')}\n"

            # 格式化pagemap内容
            pagemap_info = {k: v for k, v in info.items() if k not in ['title', 'link', 'snippet']}
            for key, value in pagemap_info.items():
                formatted_str += f"{key}: {value}\n"

            formatted_info.append(formatted_str)
        return "\n".join(formatted_info)

    def generate_answer(self, item, ocassion, purpose, match):
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = ("You are a usage location assistant, and you are very good at summarizing the circumstances "
                      "under which an item might be needed based on its attributes and description, as well as online search results.")
        prompt = (
            f"Based on the following search results, provides an insight into the target item: \"{item}\".\n"
            f"Search Results for wearing occasions:\n{ocassion}\n"
            f"Search Results for wearing purposes:\n{purpose}\n"
            f"Search results for clothing matching:\n{match}\n"
            "You need to focus on the use of this product that may be different from similar products. "
            "You should limit the output in 30 words. And don't include any detailed item attribute information. "
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages)
        return response

    def execute_task(self, item):  # 纯联网搜索
        query_occasion = f"What occasions are suitable for wearing {item}?"
        query_purpose = f"What is the purpose of wearing {item}?"
        query_match = f"What clothing or accessories are suitable to wear with {item}?"

        occasion_results = self.google_search(query_occasion)
        purpose_results = self.google_search(query_purpose)
        match_results = self.google_search(query_match)

        answer = self.generate_answer(item, occasion_results, purpose_results, match_results)
        return answer

    def execute_task1(self, item):  # 纯大模型判断
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = ("You are a usage location assistant, and you are very good at summarizing the circumstances "
                      "under which an item might be needed based on its attributes and description")
        prompt = (
            f"The item information is: {item}\n"
            "You need to think step-by-step along the following lines:"
            "First, you need to extract the descriptive features of the item."
            # "For example, \'stylish women's purple Nine West Donwon sandals\' can be decomposed into "
            # "\'purple sandals\' and \'stylish Nine West Donwon sandals\'"
            # "You need to take into account every feature of the item as much as possible."
            "Second, for each descriptive feature, you should think its possible occasions and uses. "  # dress code and outfit coordination
            # "For example, \'purple sandals\' is a color feature, purple matches black, silver, and pink clothes. "
            # "Purple represents nobility and is suitable for banquets and other occasions."
            # "\'stylish Nine West Donwon sandals\' is a style feature, "
            # "stylish scandals are very popular among young people's activities such as parties and outdoor activities."
            "Last, you can summarize the occasions and uses. You only need to output the summary"
            # "For example, for this item, output: "
            # "The item is suitable for banquets, parties and outdoor activities; "
            # "This item is suitable to be matched with black, silver, pink clothes."
            "You need to focus on the use of this product that may be different from similar products. "
            "But at the same time it should include as wide a range as possible. "
            "You should limit the output in 30 words."
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages)
        return response

    def generate_knowledge(self, attribute, ocassion, purpose, match):
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = ("You are a usage location assistant, and you are very good at summarizing the circumstances "
                      "under which an item with specific attributes might be needed based on online search results.")
        prompt = (
            f"Based on the following search results, provides an insight into the target item attribute: \"{attribute}\".\n"
            f"Search Results for wearing occasions:\n{ocassion}\n"
            f"Search Results for wearing purposes:\n{purpose}\n"
            f"Search results for clothing matching:\n{match}\n"
            "You must focus on the usage of this attribute that may be different from other similar attributes. "
            "You should output fewer than ten phrases describing the most significant possible uses of this attribute, "
            "each phrase that represents only one usage has one to three words. "
            "Phrases are separated by ',', do not include any extra characters!"
            "If there are fewer than ten usages specific to this attribute, you do not need to output more than ten."
            "If 10 words are not enough to describe the main usage, do not exceed 15 words"
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages)
        return response

    def generate_knowledge_beauty(self, attribute, ocassion, purpose, match):
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = ("You are a usage location assistant, and you are very good at summarizing the circumstances "
                      "under which an item with specific attributes might be needed based on online search results.")
        prompt = (
            f"Based on the following search results, provides an insight into the target item attribute: \"{attribute}\".\n"
            f"Search Results for using conditions:\n{ocassion}\n"
            f"Search Results for using benefits:\n{purpose}\n"
            f"Search results for Compatible products:\n{match}\n"
            "You must focus on the usage of this attribute that may be different from other similar attributes. "
            "You should output fewer than fifteen phrases describing the most significant possible uses of this attribute, "
            "each phrase that represents only one usage has one to three words. "
            "Phrases are separated by ',', do not include any extra characters!"
            "If there are fewer than 15 usages specific to this attribute, you do not need to output more than 15."
            "If 15 words are not enough to describe the main usage, do not exceed 20 words"
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages)
        return response

    def generate_knowledge_music(self, attribute, ocassion, purpose, match):
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = ("You are a music analyzing assistant, and you are very good at summarizing the circumstances "
                      "under which an music with specific attributes might be needed based on online search results.")
        prompt = (
            f"Based on the following search results, provides an insight into the target music attribute: \"{attribute}\".\n"
            f"Search Results for listening occasions:\n{ocassion}\n"
            f"Search Results for listening purpose:\n{purpose}\n"
            f"Search results for Compatible musics:\n{match}\n"
            "You must focus on the usage of this attribute that may be different from other similar attributes. "
            "You should output fewer than fifteen phrases describing the most significant possible uses of this attribute, "
            "each phrase that represents only one usage has one to three words. "
            "Phrases are separated by ',', do not include any extra characters!"
            "If there are fewer than 15 usages specific to this attribute, you do not need to output more than 15."
            "If 15 words are not enough to describe the main usage, do not exceed 20 words"
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages)
        return response

    def execute_search(self, item):
        query_occasion = f"What occasions are suitable for wearing {item} clothing?"
        query_purpose = f"What is the purpose of wearing {item} clothing?"
        query_match = f"What clothing are suitable to wear with {item} clothing?"

        occasion_results = self.google_search(query_occasion)
        purpose_results = self.google_search(query_purpose)
        match_results = self.google_search(query_match)

        answer = self.generate_knowledge(item, occasion_results, purpose_results, match_results)
        return answer

    def execute_search_beauty(self, item):
        query_occasion = f"In what situations or conditions should beauty or personal care products with {item} attribute be buyed and used?"
        query_purpose = f"What are the key benefits of buying and using beauty or personal care products with {item} attribute?"
        query_match = f"Which other products can be used together with beauty or personal care products with {item} attribute?"

        occasion_results = self.google_search(query_occasion)
        purpose_results = self.google_search(query_purpose)
        match_results = self.google_search(query_match)

        answer = self.generate_knowledge_beauty(item, occasion_results, purpose_results, match_results)
        return answer

    def execute_search_music(self, item):
        query_occasion = f"In what occasions is it appropriate to listen to music with attribute {item}?"
        query_purpose = f"What is the possible purpose of listening to music with attribute {item}?"
        query_match = f"What kind of songs are suitable to be listened to with music with attribute {item}?"

        occasion_results = self.google_search(query_occasion)
        purpose_results = self.google_search(query_purpose)
        match_results = self.google_search(query_match)

        answer = self.generate_knowledge_music(item, occasion_results, purpose_results, match_results)
        return answer

    def compute_similarity(self, query, texts, tokenizer, model, batch_size=128, device='cuda'):
        all_scores = []
        model.to(device)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            pairs = [[query, text] for text in batch_texts]
            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
                scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
                all_scores.extend(scores.cpu().numpy())
        return all_scores

    def music_search_knowledge(self, input_string, csv_file_path, top_n=5):
        categories_list = input_string.split(' | ')
        # # 读取CSV文件
        # model_checkpoint = 'E:/Codes/TAIRA/multi_agent/bge-reranker-base'
        # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        # model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        # model.eval()
        df = pd.read_csv(csv_file_path)
        # # 获取第一列内容
        # texts = df['attribute'].tolist()
        # # 计算相似度
        # scores = self.compute_similarity(input_string, texts, tokenizer, model)
        # # 添加分数到DataFrame中
        # df['similarity_score'] = scores
        # # 根据分数排序并选择前N行
        # top_matches = df.nlargest(top_n, 'similarity_score')
        # result_list = [f"Possible uses of \"{row['attribute']}\"are:{row['usage']}" for _, row in top_matches.iterrows()]
        result_list = []
        for category in categories_list:
            purpose = df[df['attribute'] == category]['usage'].tolist()[0]
            result_list.append(f"Possible uses of \"{category}\"are:{purpose}\n")
        result_string = '\n'.join(result_list)
        # # 返回第二列的内容
        # top_attributes = top_matches['usage'].tolist()
        # # 生成结果字符串
        # result_string = ', '.join(top_attributes)
        # print(result_string)

        # sys_prompt = (
        #     "You are a searcher agent and you excel at summarizing unknown knowledge from search results"
        # )
        # prompt = (
        #     f"Based on the following search results, provides an insight into the target requirement: \"{input_string}\".\n"
        #     f"Search Results:\n{result_string}\n"
        #     "Please select as many as possible attributes that meet target requirements from the search results and "
        #     "exclude irrelevant attributes that conflict with product type or do not meet requirement"
        #     "and generate a insight based only on the content of the search results."
        #     "You should only keep things related to the target demand product, for example, "
        #     "If the target describes a coat, you should only keep attributes that "
        #     "describe a coat and remove things related to hats or other clothing types."
        #     "Output only one sentence containing all the selected attributes."
        #     # "The output should only contain specific descriptions."
        #     # "Output should not be longer than one sentence"
        # )
        # messages = [{"role": "system",
        #              "content": sys_prompt},
        #             {"role": "user",
        #              "content": prompt}]
        # response = get_completion(messages)
        return result_string

    def search_knowledge(self, input_string, csv_file_path, top_n=5):
        # # 读取CSV文件
        model_checkpoint = 'E:/Codes/TAIRA/multi_agent/bge-reranker-base'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        model.eval()
        df = pd.read_csv(csv_file_path)
        # 获取第一列内容
        texts = df['attribute'].tolist()
        # 计算相似度
        scores = self.compute_similarity(input_string, texts, tokenizer, model)
        # 添加分数到DataFrame中
        df['similarity_score'] = scores
        # 根据分数排序并选择前N行
        top_matches = df.nlargest(top_n, 'similarity_score')
        result_list = [f"Possible uses of \"{row['attribute']}\"are:{row['usage']}" for _, row in top_matches.iterrows()]
        # result_list = []
        # for category in categories_list:
        #     purpose = df[df['attribute'] == category]['usage'][0]
        #     result_list.append(f"Possible uses of \"{category}\"are:{purpose}\n")
        result_string = '\n'.join(result_list)
        # # 返回第二列的内容
        # top_attributes = top_matches['usage'].tolist()
        # # 生成结果字符串
        # result_string = ', '.join(top_attributes)
        # print(result_string)

        # sys_prompt = (
        #     "You are a searcher agent and you excel at summarizing unknown knowledge from search results"
        # )
        # prompt = (
        #     f"Based on the following search results, provides an insight into the target requirement: \"{input_string}\".\n"
        #     f"Search Results:\n{result_string}\n"
        #     "Please select as many as possible attributes that meet target requirements from the search results and "
        #     "exclude irrelevant attributes that conflict with product type or do not meet requirement"
        #     "and generate a insight based only on the content of the search results."
        #     "You should only keep things related to the target demand product, for example, "
        #     "If the target describes a coat, you should only keep attributes that "
        #     "describe a coat and remove things related to hats or other clothing types."
        #     "Output only one sentence containing all the selected attributes."
        #     # "The output should only contain specific descriptions."
        #     # "Output should not be longer than one sentence"
        # )
        # messages = [{"role": "system",
        #              "content": sys_prompt},
        #             {"role": "user",
        #              "content": prompt}]
        # response = get_completion(messages)
        return result_string

