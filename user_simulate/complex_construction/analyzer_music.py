import random

import pandas as pd
import torch

from agent import Agent
from complex_query.utils.task import get_completion
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from searcher import SearcherAgent


class AnalyzeAgent(Agent):
    def __init__(self):
        super().__init__("ChatAgent")
        self.query_examples = """Can you recommend me a [product] for [requirement]?
                I am looking for a [product] that is [requirement].
                Do you have any suggestions for a [product] that would be good for [requirement]?
                I'd like to find a [product] that is [requirement].
                Can you help me choose a [product] for [requirement]?
                What would you recommend for a [product] that meets the [requirement]?
                I'm interested in a [product] that can help me with [requirement].
                Could you suggest a [product] that fits my [requirement]?
                I want to find a [product] that has [requirement].
                Show me a [product] that works well for [requirement].
                I need a [product] that is [requirement].
                What would be the best [product] for [requirement]?
                I’m searching for a [product] that’s [requirement].
                Do you have a [product] for [requirement]?
                I want a [product] that is [requirement] and [requirement].
                Could you recommend a [product] that fits [requirement]?
                I’m looking for a [product] that has [requirement].
                Please suggest a [product] for [requirement].
                I need something like a [product] that’s good for [requirement].
                Do you know any [product] that would work well for [requirement]?
                Help me find a [product] that can handle [requirement].
                I'm after a [product] that works best for [requirement].
                What are the best [product] options for [requirement]?
                I’d like a [product] that’s specifically designed for [requirement].
                Could you guide me to a [product] that is best suited for [requirement]?
                I want a [product] that’s [requirement] and reliable.
                Can you help me pick a [product] for [requirement]?
                I am interested in a [product] that is [requirement] and [requirement].
                I’m in need of a [product] that is good for [requirement] and [requirement].
                Do you have any suggestions for a [product] with [requirement]?
                What’s the best [product] I can use for [requirement]?
                I need a [product] that’s perfect for [requirement] and [requirement].
                I’m looking for something that’s a [product] and also works for [requirement].
                I’d like to buy a [product] with [requirement] in mind.
                Show me a [product] that will solve my [requirement].
                I want a [product] that has both [requirement] and [requirement].
                Can you suggest a [product] that will help with [requirement]?
                I’m looking for a [product] that provides [requirement].
                Can you point me to a [product] that’s best for [requirement]?
                I’d like a [product] that’s great for [requirement].
                I want a [product] that can meet my [requirement] and [requirement].
                Can you recommend a [product] that will work for [requirement]?
                I need a [product] that is [requirement] and easy to use.
                Can you show me a [product] that works for [requirement]?
                I want to try a [product] that’s [requirement] and affordable.""".split('\n')
        self.item_query_examples = """Can you suggest a [product] for me?
            I’m looking for a [product].
            Do you know any good [product]?
            I need a [product].
            Can you recommend a [product]?
            I want to find a [product].
            What is the best [product]?
            Could you tell me which [product] is best?
            I’m interested in a [product].
            Could you suggest a [product]?
            I need advice on choosing a [product].
            Are there any [product] that is great?
            Can you help me pick a [product]?
            I'm searching for a [product].
            Can you recommend a [product] for me to try?
            What [product] do you recommend?
            Could you help me find a [product]?
            What’s a good [product] to try?
            Any suggestions for a [product]?
            What are some options for a [product]?
            Can you point me to a [product] I might like?
            I need a [product], what do you suggest?
            What’s the best [product] available?
            Where can I find a [product]?
            Can you show me a [product]?
            What [product] should I go for?
            I’m looking for a [product], any ideas?
            What is a great [product]?
            Do you have any [product] to recommend?
            What kind of [product] would you suggest?""".split('\n')

    def extract_item(self, item, category):
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = ("You are a music product analyst, "
                      "you are good at summarizing the attributes and characteristics of products")
        prompt = f"The music is:{item}."\
                 f"The description and classification of the music are: \n\"{category}\".\n" \
                 f"You should keep as much information as possible about this part." \
                 f"For each word in this section, you should consider whether it can be included in the description of the music." \
                 "Please use one descriptive noun phrases of 20 words or less as output." \
                 "Don't include descriptions that don't appear in the original product description."
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages)
        # print(f"Chat Response: {response}")
        return response

    def generate_query(self, item, item_info, cat_number, user):
        random_queries = '\n'.join(random.sample(self.query_examples, 3))
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = (
            "You are a troublesome music chooser. You are good at asking complex queries that "
            "conversational recommendation systems cannot easily retrieve."
            # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
        )
        prompt = (
            f"The music description is {item}"
            f"The possible uses of the intended music are described as follows: \n\"{item_info}\".\n"
            "Based on this, please construct one user queries in the conversation recommendation, "
            "where the target music item point to the music described above. "
            "For each particular of the item, please refer to the possible uses to translate it into an actual need."
            "and it should be the same as the item description. "
            f"Also, the user's base information should be added to the query as a constraint if possible: {user}"
            "For example, gender and age."  # 衣服特有
            "In addition to the above rules, you shouldn't include words "
            "as well as descriptive descriptions that are already in the item description, "  # 难度语句
            # "as well as any descriptive descriptions (such as comfortable, sports, thermal, etc)"
            "because these descriptive descriptions are easy to retrieve in the database, "
            "but you will need to include them in the form of a more realistic need for the query "
            "(Such as changed to play in a party, listen on the way to work, for sweet date, etc.)"
            "In addition, you can **only specify one requirement at a query**, and cannot add 'and' to specify multiple requirements in a query."
            # "You must try not to include words in the converted description that are included in the original "
            # "item information."
            "\n"
            ""
            "Here are some examples:\n"
            "Can you suggest a music for a relaxing drive to work?"
            "I need a upbeat tune for my morning workout. Any ideas?"
            "Can you recommend a calming music for when I need to focus on work?"
            "I'm looking for a chill background music for a dinner party. Any recommendations?"
            "Please suggest a energetic track for a road trip with friends."
            "I'm having a quiet night in. Can you recommend something peaceful to listen to?"
            "What music would be great for a casual hangout with friends?"
            "I'm hosting a get-together and need some background music. Any suggestions?"
            "I’m looking for music to help me unwind after a stressful day. Any recommendations?"
            "Can you suggest a tune for a fun afternoon while doing chores at home?"
            # "No longer than 1 sentences."
            f"You must choose the most appropriate template from these to construct the query:{random_queries}"
            "If the template does not fit your needs, you can modify individual words to make the sentence natural and real."
            "Your query needs to be realistic because it is a real need. "
            "Do not include statements that may cause ambiguity. "
            # " or 25 words"
        )
        if cat_number >= 5:
            prompt += "The total length should not exceed 2 sentences or 25 words"
        else:
            prompt += "The total length should not exceed 1 sentences"
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        # response = get_completion(messages, llm="gpt-4o-mini")
        response = get_completion(messages, llm="gpt-4o")
        # print(f"Chat Response: {response}")
        return response

    def item_query(self, item, cat_number, user):
        random_queries = '\n'.join(random.sample(self.item_query_examples, 3))
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = (
            "You are a online shopper. You are seeking for a certain product."
            # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
        )
        prompt = (
            f"The music description is {item}"
            "Based on this, please construct one user queries in the conversation recommendation, "
            "where the target music item point to the music described above. "
            f"Also, the user's base information should be added to the query as a constraint if possible: {user}"
            "For example, gender and age."  # 衣服特有
            # "You must try not to include words in the converted description that are included in the original "
            # "item information."
            f"You must choose the most appropriate template from these to construct the query:{random_queries}"
            "If the template does not fit your needs, you can modify individual words to make the sentence natural and real."
            "Your query needs to be realistic because it is a real need. "
            "Do not include statements that may cause ambiguity. "

            # " or 25 words"
        )
        if cat_number >= 3:
            prompt += "The total length should not exceed 2 sentences or 25 words"
        else:
            prompt += "The total length should not exceed 1 sentences"
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages, llm="gpt-4o")
        # print(f"Chat Response: {response}")
        return response

    def classify_query(self, query, item, number):
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = (
            "You are a troublesome shopper. You are good at rewriting queries in conversational recommendations to be more complex."
            # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
        )
        prompt = (
            "The query contains the user's specific requirement, and the item description contains the attributes of the music."
            "There are several ways to rewrite the query."
            "(1) If the requirement in the query can be satisfied by different music styles, you can request 'multiple different styles' of musics.\n"
            "(2) If one of the music attributes can satisfy multiple sceneries, you can request more sceneries.\n"
            "(3) If the query is not particularly specific (For example, "
            "the need for dinner is not a specific requirement, "
            "Because the dinner scene can be a formal dinner or a gathering with friends. "
            "This method should be given a little higher priority, but be sure there is no conflict against the condition \n"
            f"The original query is {query}"
            f"The item description is {item}"
            f"The assigned number is {number}"
            "Please determine whether this query may conflict with the conditions of this method."
            "If there is no conflict, then still choose this number. Otherwise, output the number you think is most suitable."
            "You should prioritize the number you are assigned, but do not keep unreasonable classifications."
            "output the number in an '{}', '{}' contains pure number."
            "You must output a number!"
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages, llm="gpt-4o-mini")
        # print(f"Chat Response: {response}")
        return response

    def rewrite_query(self, query, item, number, df, model=None):
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = (
            "You are a music chooser. You are good at rewriting queries in conversational recommendations."
            # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
        )
        prompt = (
            f"The original query is {query}"
            # f"The item description is {item}"
        )
        if number == '1':
            prompt = prompt + (
                "The requirement in the query can be satisfied by different music styles. "
                "Please add 'Recommend me some different styles of music' after the original query. "
                "For example, rewrite 'What’s the best music I can use for a retro-themed party with a vibrant and energetic atmosphere?' "
                "to 'What’s the best music I can use for a retro-themed party with a vibrant and energetic atmosphere? "
                "Please recommend me some different styles of music'"
                # "Additionally, keep the description of men's, women's, boy's or girl's. "
            )
        elif number == '2':
            searcher = SearcherAgent()
            more_requirements = searcher.search_knowledge(item, '../knowledge/music_knowledge0.csv')
            prompt = prompt + (
                "One of the music attributes can satisfy multiple sceneries. "
                f"Here are some more scenarios where this music might be used: {more_requirements}"
                f"Please select a music attribute associated with the original query, and select one or two of the "
                f"above scenarios to add to the original query, and request recommendations for each."
                "For example: Rewrite 'I’m looking for something that’s a classic country collection and also works for a nostalgic family gathering.' "
                "into 'I really enjoy classical country music, and I’m looking for different pieces of music for specific occasions. "
                "One for a nostalgic family gathering, one for reading, and one for bathing.'"
            )

        elif number == '3':
            prompt = prompt + (
                "The query should be not particularly specific, "
                "Please change the query requirement to be a little less detailed and "
                "add this sentence at the end: ' I'm not sure about the specific listening scene'. "
            )
        prompt = prompt + (
            "You cannot make major changes to the original query, such as adding new demand scenarios or "
            "replacing existing demand scenarios or add descriptive words for the target product. "
            "You can ONLY make appropriate additions or modifications to the query **STRICTLY** according to the instructions above!"
            "Please remember that the rewritten query should still be asked in the user's tone of voice just like the original one."
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        if model is None:
            response = get_completion(messages, 'gpt-4o-mini')
        else:
            response = get_completion(messages, model)
        print(number)
        print(prompt)
        return response

    def count_rec_number(self, query):
        sys_prompt = "You are a conversational recommendation user query understanding assistant."
        prompt = (
            f"User query is: \"{query}\"\n"
            "Please count how many types of products the user has requested in this query. "
            "Output a json format to include the quantity and specific products types of the required products."
            "For example: 'I'm looking for something that's East Coast Rap & Hip-Hop and also works for a lively city night out. "
            "I’d love recommendations that also fit a party atmosphere and community connection.' "
            "This query raises 3 requirements: lively city night out, a party atmosphere and community connection."
            "And you should output this json format:"
            "{\n"
            "  \"product_count\": 3,\n"
            "  \"product_list\": [\n"
            "    \"merchandising tags without string\",\n"
            "    \"tagging gun\",\n"
            "    \"unstrung tags\"\n"
            "  ]\n"
            "}\n"
            "In particular, when the recommendation requirement just contains the expression 'a set' which indicates that "
            "the number of types for recommendations is unclear, you should output {\"product_count\": -1} to indicate that the number is uncertain."
            "Please reflect the literal expression of the user's query, and do not make decisions for users about "
            "products that are not mentioned. Whenever 'a set' appears, you should output -1."
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        response = get_completion(messages)
        return response.strip()

    def get_top_n_similar_queries(self, user_query, df, batch_size=128, device='cuda', top_k=10):
        # 提取 DataFrame 中的所有查询
        project_texts = df['query'].tolist()
        model_checkpoint = 'E:/Codes/TAIRA/multi_agent/bge-reranker-base'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        model.eval()

        # 计算相似度
        similarity_scores = self.compute_similarity_with_reranker(user_query, project_texts, tokenizer, model,
                                                                  batch_size=batch_size, device=device)

        # 将相似度分数添加到 DataFrame
        df['similarity_score'] = similarity_scores

        # 获取相似度最高的 top_k 行
        top_k_projects = df.nlargest(top_k, 'similarity_score')

        return top_k_projects

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


    def examine_query(self, query):
        sys_prompt = "You are a conversational recommendation user query understanding assistant."
        prompt = (
            f"Please check if there are multiple requirements in the original query: {query}. "
            f"If there are, you need to remove the redundant ones and keep only one requirement. "
            f"For example: 'for themed gatherings and creative expression.' is a multiple requirement, "
            f"you can remove the 'creative expression'."
            "You only need to output the revised query, with nothing else!"
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        response = get_completion(messages)
        return response.strip()

    def examine_item(self, query):
        sys_prompt = "You are a conversational recommendation user query understanding assistant."
        prompt = (
            f"Please check if there is description of the scene requirements in the original query: {query}. "
            f"If there are, you need to remove the description of the scene requirements and keep only item attributes. "
            f"For example: 'that are comfortable and suitable for daily use.' is a description of the scene requirements, "
            f"you should change it to simply 'that are comfortable'."
            f"In addition, keep the query concise, retaining only "
            f"the description of the target music attributes and nothing else."
            "You only need to output the revised query, with nothing else!"
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        response = get_completion(messages)
        return response.strip()

    def execute_task(self, task):
        return


