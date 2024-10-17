import os

import numpy as np

from .agent import Agent
from FlagEmbedding import BGEM3FlagModel

import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.Prompts import CLOTH_RETRIEVE_PROMPT, PRODUCT_RETRIEVE_PROMPT, BEAUTY_RETRIEVE_PROMPT


def compute_similarity(query_embedding, project_embeddings):
    similarity_scores = query_embedding @ project_embeddings.T
    return similarity_scores


class ItemRetrievalAgent(Agent):
    def __init__(self, memory):
        super().__init__("ItemRetrievalAgent", memory)
        with open('system_config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.domain = self.config['DOMAIN']
        self.domain_path = "data/" + self.domain
        if self.domain == "amazon_clothing":
            self.df = pd.read_parquet(self.domain_path + '/metadata.parquet')
            self.df['project_info'] = self.df['title'] + ' ' + self.df['categories']
            self.df.rename(columns={'id': 'product_id'}, inplace=True)
        elif self.domain == "amazon_beauty":
            self.df = pd.read_parquet(self.domain_path + '/metadata.parquet')
            self.df['title'] = self.df['title'].fillna('')
            self.df['description'] = self.df['description'].fillna('')
            self.df['categories'] = self.df['categories'].fillna('')

            self.df['title'] = self.df['title'].astype(str)
            self.df['description'] = self.df['description'].astype(str)
            self.df['categories'] = self.df['categories'].astype(str)
            self.df['project_info'] = self.df['title'] + ' ' + self.df['description'] + self.df['categories']
            self.df.rename(columns={'asin': 'product_id'}, inplace=True)
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
            model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            project_texts = self.df['project_info'].tolist()
            # Debugging information
            for text in project_texts:
                # print(text)
                if not isinstance(text, str):
                    print(f"Non-string item found: {text}")

            self.project_embeddings = model.encode(project_texts, batch_size=64, max_length=8192)['dense_vecs']
            np.save(self.embedding_file, self.project_embeddings)

    def parse_user_input(self, user_input):
        history = self.memory.get_history()
        history_str = history
        sys_prompt = "You're a recommendation assistant and you're good at recognizing user preferences."
        if self.domain == "amazon_clothing":
            prompt = CLOTH_RETRIEVE_PROMPT.replace('{user_input}', user_input)
        elif self.domain == "amazon_beauty":
            prompt = BEAUTY_RETRIEVE_PROMPT.replace('{user_input}', user_input)
        elif self.domain == "kdd22":
            prompt = PRODUCT_RETRIEVE_PROMPT.replace('{user_input}', user_input)

        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}]
        response = self.call_gpt(messages, llm='gpt-4o-mini')
        print(response)
        structured_preferences = response
        return structured_preferences

    def recommend_projects_with_bge_m3(self, user_query, top_n=500, initial_filter=100):
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        base_type = user_query.split(";")[0].strip()

        base_type_embedding = model.encode([base_type], batch_size=1, max_length=8192)['dense_vecs']

        initial_similarity_scores = compute_similarity(base_type_embedding, self.project_embeddings)[0]
        initial_similarity_scores = np.array(initial_similarity_scores, dtype=np.float32)

        self.df['initial_similarity_score'] = initial_similarity_scores
        filtered_df = self.df.nlargest(initial_filter, 'initial_similarity_score')

        project_texts = filtered_df['project_info'].tolist()

        query_embedding = model.encode([user_query], batch_size=1, max_length=8192)['dense_vecs']

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

        top_n_projects['similarity_score'] = similarity_scores
        top_k_projects = top_n_projects.nlargest(top_k, 'similarity_score')
        return top_k_projects

    def execute_task(self, query):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        top_k = self.config['TOPK_ITEMS']
        top_n = self.config['TOPN_ITEMS']
        reference = self.parse_user_input(query)


        top_n_projects = self.recommend_projects_with_bge_m3(reference, top_n)
        top_k_projects = self.recommend_projects_with_reranker(reference, top_n_projects, top_k, device=device)
        top_k_projects['project_info'] = top_k_projects['project_info'].apply(lambda x: x[:800] if len(x) > 800 else x)
        return top_k_projects[['product_id', 'project_info']]
