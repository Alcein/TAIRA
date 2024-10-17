# executor_agents/searcher.py
import json
import os
import urllib.parse

import pandas as pd

from .agent import Agent
import requests
from utils.task import get_completion
from utils.task import extract_braces_content

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SearcherAgent(Agent):
    def __init__(self, memory):
        super().__init__("SearcherAgent", memory)
        with open('system_config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.domain = self.config['DOMAIN']
        self.domain_path = "data/" + self.domain
        self.api_key = ''
        self.cse_id = ''

    def google_search(self, query, num_results=5):
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.cse_id}&q={encoded_query}"
        response = requests.get(url)
        response.raise_for_status()
        results = response.json().get('items', [])
        return self.extract_and_format_info(results[:num_results])

    def extract_and_format_info(self, results):
        useful_info = []
        for item in results:
            info = {
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', '')
            }

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

            pagemap_info = {k: v for k, v in info.items() if k not in ['title', 'link', 'snippet']}
            for key, value in pagemap_info.items():
                formatted_str += f"{key}: {value}\n"

            formatted_info.append(formatted_str)
        return "\n".join(formatted_info)

    def generate_answer(self, query):
        context = self.google_search(query)
        history = self.memory.get_history()
        history_str = history
        sys_prompt = ("You are a searcher agent and you excel at acquiring previously unknown knowledge through search results"
                      f"Here is the previous conversation history:\n{history_str}\n")
        prompt = (
            f"Based on the following search results, provides an insight into the target query: \"{query}\".\n"
            f"Search Results:\n{context}\n"
            "Insight:"
            "At last, output an summary in JSON format:\n"
            "{\"summary\": \"...\"}"
            "The output should only contain specific descriptions."
            "The output is a keyword combination of no more than 20 words, not a descriptive sentence."
            "You should give a specific answer to the question."
            "Please **strictly** follow this json format output"
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages)
        return response

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

    def generate_knowledge(self, input_string, csv_file_path, top_n=20):
        answer = self.generate_answer(input_string)
        summary = json.loads(extract_braces_content(answer))["summary"]
        if os.path.exists(csv_file_path):
            model_checkpoint = 'bge-reranker-base'
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
            model.eval()
            df = pd.read_parquet(csv_file_path)
            texts = (df['attribute'] + ',' + df['usage']).tolist()
            scores = self.compute_similarity(summary, texts, tokenizer, model)
            df['similarity_score'] = scores
            top_matches = df.nlargest(top_n, 'similarity_score')
            top_attributes = top_matches['attribute'].tolist()
            result_string = ', '.join(top_attributes)
        else:
            result_string = summary
        sys_prompt = (
            "You are a searcher agent and you excel at summarizing unknown knowledge from search results"
        )
        prompt = (
            f"Based on the following search results, provides an insight into the target requirement: \"{input_string}\".\n"
            f"Search Results:\n{result_string}\n"
            "Please select as many as possible attributes that meet target requirements from the search results and "
            "exclude irrelevant attributes that conflict with product type or do not meet requirement"
            "and generate a insight based only on the content of the search results."
            "You should only keep things related to the target demand product, for example, "
            "If the target describes a coat, you should only keep attributes that "
            "describe a coat and remove things related to hats or other clothing types."
            "You can only generate your output based on the words in the Search Results."
            "Don't include content that is not in the search results, even if you think it is in line with the needs"
            "Output one sentence insight containing all the selected attributes. Don't include too little information."
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages)
        return response

    def execute_task(self, task):
        query = task
        knowledge_file = self.domain_path+"/knowledge1.parquet"
        knowledge = self.generate_knowledge(query, knowledge_file)
        return knowledge
