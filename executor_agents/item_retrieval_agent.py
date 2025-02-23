# agents/item_retrieval_agent.py
import os

import numpy as np
from fuzzywuzzy import fuzz

from .agent import Agent
from FlagEmbedding import BGEM3FlagModel

import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.Prompts import CLOTH_RETRIEVE_PROMPT, PRODUCT_RETRIEVE_PROMPT, BEAUTY_RETRIEVE_PROMPT, MUSIC_RETRIEVE_PROMPT
from utils.memory import Memory
# import openai
from openai import OpenAI

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize


def compute_similarity(query_embedding, project_embeddings):
    similarity_scores = query_embedding @ project_embeddings.T
    return similarity_scores


class ItemRetrievalAgent(Agent):
    def __init__(self, memory):
        # print(1)
        super().__init__("ItemRetrievalAgent", memory)
        with open('system_config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.domain = self.config['DOMAIN']
        self.domain_path = "data/" + self.domain
        # print(1)
        if self.domain == "amazon_clothing":
            self.df = pd.read_csv(self.domain_path + '/metadata.csv')
            self.df['project_info'] = self.df['title'] + ' ' + self.df['category']
            self.df.rename(columns={'id': 'product_id'}, inplace=True)
        elif self.domain == "amazon_music":
            self.df = pd.read_csv(self.domain_path + '/metadata.csv')
            self.df['project_info'] = self.df['title'] + ' ' + self.df['category']
            self.df.rename(columns={'id': 'product_id'}, inplace=True)
        elif self.domain == "amazon_beauty":
            self.df = pd.read_csv(self.domain_path + '/metadata.csv')
            self.df['title'] = self.df['title'].fillna('')
            self.df['description'] = self.df['description'].fillna('')
            self.df['category'] = self.df['category'].fillna('')

            # 确保将所有内容转为字符串
            self.df['title'] = self.df['title'].astype(str)
            self.df['description'] = self.df['description'].astype(str)
            self.df['category'] = self.df['category'].astype(str)
            self.df['project_info'] = self.df['title'] + ' ' + self.df['description'] + self.df['category']
            self.df.rename(columns={'id': 'product_id'}, inplace=True)
        elif self.domain == "kdd22":
            self.df = pd.read_parquet(self.domain_path + '/shopping_queries_dataset_products.parquet')
            self.df = self.df[self.df['product_locale'] == 'us']
            self.df['project_info'] = self.df['product_title'] + ' ' + self.df['product_description']
            self.df = self.df.dropna(subset=['project_info'])
            self.df = self.df.reset_index(drop=True)

        self.embedding_file = os.path.join(self.domain_path, 'project_embeddings.npy')

        # Load embedding
        if os.path.exists(self.embedding_file):
            self.project_embeddings = np.load(self.embedding_file)
        else:
            model = BGEM3FlagModel('../multi_agent/bge-m3', use_fp16=True)
            project_texts = self.df['project_info'].tolist()
            # Debugging information
            for text in project_texts:
                # print(text)
                if not isinstance(text, str):
                    print(f"Non-string item found: {text}")

            self.project_embeddings = model.encode(project_texts, batch_size=64, max_length=8192)['dense_vecs']
            np.save(self.embedding_file, self.project_embeddings)
        self.item_df =self.df
        self.projects = self.df
        self.corpus = self.projects['project_info'].astype(str).tolist()
        self.preference = ''
        # 使用 BM25 模型
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("complete bm25")

    def parse_user_input(self, user_input):
        history = self.memory.get_history()
        history_str = history
        sys_prompt = "You're a recommendation assistant and you're good at recognizing user preferences."
        prompt = f"The user's personalized preferences are: {self.preference}"
        if self.domain == "amazon_clothing":
            prompt += CLOTH_RETRIEVE_PROMPT.replace('{user_input}', user_input)
        elif self.domain == "amazon_beauty":
            prompt += BEAUTY_RETRIEVE_PROMPT.replace('{user_input}', user_input)
        elif self.domain == "amazon_music":
            prompt += MUSIC_RETRIEVE_PROMPT.replace('{user_input}', user_input)

        # prompt = "The user's query is:{user_input}\nFrom this, please identify the user's requirements and preferences for clothing.\nPlease fill in this format and only output the filled content:[clothing type]; [preference]. Separate multiple attributes with ' '.\nYou can expand on the user's statement as you see fit, but don't make inferences that aren't there in the user's meaning. If the user has no requirement for an attribute, please fill the corresponding position with ' '."
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}]
        # response = get_completion(messages, llm='gpt-4o-mini')
        response = self.call_gpt(messages, llm='gpt-4o-mini')
        print(response)
        structured_preferences = response
        return structured_preferences

    def recommend_projects_with_bge_m3(self, user_query, top_n=1000, initial_filter=1000):
        model = BGEM3FlagModel('../multi_agent/bge-m3', use_fp16=True)
        # if self.domain == 'amazon_clothing':
        # 从用户查询中提取 base_type
        base_type = user_query.split(";")[0].strip()
        # print('base_type:', base_type)

        # 对 base_type 进行编码
        base_type_embedding = model.encode([base_type], batch_size=1, max_length=8192)['dense_vecs']

        # 计算 base_type 与所有项目的相似度
        initial_similarity_scores = compute_similarity(base_type_embedding, self.project_embeddings)[0]
        initial_similarity_scores = np.array(initial_similarity_scores, dtype=np.float32)

        # 初步筛选出相似度最高的 initial_filter 项
        self.df['initial_similarity_score'] = initial_similarity_scores
        filtered_df = self.df.nlargest(top_n, 'initial_similarity_score')
        filtered_df = filtered_df[filtered_df['initial_similarity_score'] >= 0.6]

        # # else:
        # #     filtered_df = self.df
        #
        # # 获取初步筛选的项目描述
        # project_texts = filtered_df['project_info'].tolist()
        #
        # # 对用户查询进行编码
        # query_embedding = model.encode([user_query], batch_size=1, max_length=8192)['dense_vecs']
        #
        # # 计算用户查询与初步筛选项目的相似度
        # filtered_project_embeddings = self.project_embeddings[filtered_df.index]
        # final_similarity_scores = compute_similarity(query_embedding, filtered_project_embeddings)[0]
        # final_similarity_scores = np.array(final_similarity_scores, dtype=np.float32)
        #
        # filtered_df['similarity_score'] = final_similarity_scores
        # top_n_projects = filtered_df.nlargest(top_n, 'similarity_score')
        # # return top_n_projects
        return filtered_df

    def recommend_projects_with_bge_m3_base(self, user_query, top_n=500):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(device)
        model = BGEM3FlagModel('../multi_agent/bge-m3', use_fp16=True, device = device)
        query_embedding = model.encode([user_query], batch_size=1, max_length=8192)['dense_vecs']
        similarity_scores = compute_similarity(query_embedding, self.project_embeddings)[0]
        similarity_scores = np.array(similarity_scores, dtype=np.float32)

        if len(similarity_scores) != len(self.df):
            raise ValueError("Length of similarity_scores does not match length of DataFrame")

        self.df['similarity_score'] = similarity_scores
        top_n_projects = self.df.nlargest(top_n, 'similarity_score')
        return top_n_projects

    def recommend_projects_with_fuzzy(self, user_query, top_n=500, initial_filter=100):
        model = BGEM3FlagModel('../multi_agent/bge-m3', use_fp16=True)
        # if self.domain == 'amazon_clothing':
        # 从用户查询中提取 base_type
        base_type = user_query.split(";")[0].strip()
        # print('base_type:', base_type)

        # 对 base_type 进行编码
        base_type_embedding = model.encode([base_type], batch_size=1, max_length=8192)['dense_vecs']

        # 计算 base_type 与所有项目的相似度
        initial_similarity_scores = compute_similarity(base_type_embedding, self.project_embeddings)[0]
        initial_similarity_scores = np.array(initial_similarity_scores, dtype=np.float32)

        # 初步筛选出相似度最高的 initial_filter 项
        self.df['initial_similarity_score'] = initial_similarity_scores
        filtered_df = self.df.nlargest(initial_filter, 'initial_similarity_score')

        # else:
        #     filtered_df = self.df

        # 获取初步筛选的项目描述
        project_texts = filtered_df['project_info'].tolist()

        # 对用户查询进行编码
        query_embedding = model.encode([user_query], batch_size=1, max_length=8192)['dense_vecs']

        # 计算用户查询与初步筛选项目的相似度
        filtered_project_embeddings = self.project_embeddings[filtered_df.index]
        final_similarity_scores = compute_similarity(query_embedding, filtered_project_embeddings)[0]
        final_similarity_scores = np.array(final_similarity_scores, dtype=np.float32)

        filtered_df['similarity_score'] = final_similarity_scores
        top_n_projects = filtered_df.nlargest(top_n, 'similarity_score')
        return top_n_projects

    def compute_similarity_with_reranker(self, query, texts, tokenizer, model, batch_size=128, device='cuda'):
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

    def recommend_projects_with_reranker(self, user_query, top_n_projects, top_k=10, device='cpu'):
        project_texts = top_n_projects['project_info'].tolist()
        model_checkpoint = 'bge-reranker-base'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        model.eval()
        similarity_scores = self.compute_similarity_with_reranker(user_query, project_texts, tokenizer, model,
                                                                  device=device)

        top_n_projects['similarity_score_reranker'] = similarity_scores
        top_n_projects.to_csv('top_n_projects.csv', index=False)
        top_k_projects = top_n_projects.nlargest(top_k, 'similarity_score_reranker')
        return top_k_projects

    def recommend_projects_with_reranker_base(self, user_query, top_k=10):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        top_k = self.config['TOPK_ITEMS']
        top_n = self.config['TOPN_ITEMS']
        reference = self.parse_user_input(user_query)
        # print(reference)
        # reference = query
        # base_type = reference.split(";")[0]

        top_n_projects = self.recommend_projects_with_bge_m3(reference, top_n)
        top_k_projects = self.recommend_projects_with_reranker(reference, top_n_projects, top_k, device=device)
        top_k_projects['project_info'] = top_k_projects['project_info'].apply(lambda x: x[:800] if len(x) > 800 else x)
        return top_k_projects
    def match_projects_with_BM25(self, user_query, top_k=10):
        # 对查询进行分词
        tokenized_query = word_tokenize(user_query.lower())
        # 使用 BM25 计算相似度得分
        scores = self.bm25.get_scores(tokenized_query)
        # 获取得分最高的 top_k 个项目
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        # 根据索引获取对应的 DataFrame 行
        top_k_projects = self.projects.iloc[top_k_indices].copy()
        # 添加相似度得分列
        top_k_projects['similarity_score'] = [scores[i] for i in top_k_indices]
        return top_k_projects

    def execute_task(self, query):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        top_k = self.config['TOPK_ITEMS']
        top_n = self.config['TOPN_ITEMS']
        reference = self.parse_user_input(query)
        # print(reference)
        # reference = query
        # base_type = reference.split(";")[0]

        top_n_projects = self.recommend_projects_with_bge_m3(reference, top_n)
        top_k_projects = self.recommend_projects_with_reranker(reference, top_n_projects, top_k, device=device)
        top_k_projects['project_info'] = top_k_projects['project_info'].apply(lambda x: x[:800] if len(x) > 800 else x)
        return top_k_projects[['product_id', 'project_info']]

    def match_query(self, user_query, threshold=70):
        def compute_similarity(row):
            return fuzz.partial_ratio(user_query, row['project_info'])

        # 应用函数计算相似度并添加到新列
        self.item_df['similarity_score'] = self.item_df.apply(compute_similarity, axis=1)

        # 筛选相似度 >= 阈值的行
        filtered_df = self.item_df[self.item_df['similarity_score'] >= threshold]

        return filtered_df


