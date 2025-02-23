import random

import pandas as pd
import torch

from agent import Agent
from complex_query.utils.task import get_completion
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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

    def extract_item(self, item, category):
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = ("You are a product analyst, "
                      "you are good at summarizing the attributes and characteristics of products")
        prompt = f"The item is:{item}. You need to extract the most basic type and gender attributes " \
                 f"(such as men's suits) and discard all other information, only gender and basic type!" \
                 f"The description and classification of the items are: \n\"{category}\".\n" \
                 f"You should keep as much information as possible about this part." \
                 f"For each word in this section, you should consider whether it can be included in the description of the item." \
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
            "You are a troublesome shopper. You are good at asking complex queries that "
            "conversational recommendation systems cannot easily retrieve."
            # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
        )
        prompt = (
            f"The item description is {item}"
            f"The possible uses of the intended product are described as follows: \n\"{item_info}\".\n"
            "Based on this, please construct one user queries in the conversation recommendation, "
            "where the target item point to the item described above. "
            "For each particular of the item, please refer to the possible uses to translate it into an actual need."
            "The most basic types of products (such as dress, sandals, etc) must be specified in the query in one word, "
            "and it should be the same as the item description. "
            "In addition, if the product specifies the gender, it must be added. "  # 衣服特有
            f"Also, the user's base information should be added to the query as a constraint if possible: {user}"
            "For example, gender and body type."  # 衣服特有
            "In addition to the above three rules, you are abandoned to include any words that are already in the item description, "  # 难度语句
            "as well as any descriptive descriptions (such as comfortable, sports, thermal, etc)"
            "because these descriptive descriptions are easy to retrieve in the database, "
            "but you will need to include them in the form of a more realistic need for the query "
            "(Such as changed to skin sensitivity, basketball game, arctic circles, etc.)"
            "You must try not to include words in the converted description that are included in the original "
            "item information."
            "\n"
            ""
            "Here are some examples:\n"
            "Please recommend some suitable dresses to wear at a wedding\n"
            "Is there any suitable trousers to wear when attending a meeting?\n"
            # "I want some men T-shirts to go with gold necklaces\n"
            "Please recommend some suitable men's jacket to wear to art exhibitions\n"
            "Any recommendations for men's coat suitable for taking formal photos?\n"
            # "Please recommend some dresses that would go well with purple heels.\n"
            "I need some men's shoes for mountain biking.\n"
            "Do you have any recommendations for skirts to wear to Disneyland?\n"
            "I want some Women's Jackets that I can wear straight from a business meeting to a dinner party.\n"
            "Please recommend some women's clothing for working in a museum\n"
            f"You must choose the most appropriate template from these to construct the query:{random_queries}"
            "Your query needs to be realistic because it is a real need. "
            "Do not include statements that may cause ambiguity. "
            # "No longer than 1 sentences."

            # " or 25 words"
        )
        if cat_number >= 10:
            prompt += "The total length should not exceed 2 sentences or 25 words"
        else:
            prompt += "The total length should not exceed 1 sentences"
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages, llm="gpt-4o-mini")
        # print(f"Chat Response: {response}")
        return response

    def item_query(self, item, cat_number, user):
        random_queries = '\n'.join(random.sample(self.query_examples, 3))
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = (
            "You are a online shopper. You are seeking for a certain product."
            # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
        )
        prompt = (
            f"The item description is {item}"
            "Based on this, please construct one user queries in the conversation recommendation, "
            "where the target items point to the items described above. "
            # "For each particular of the item, please refer to the possible uses to translate it into an actual need."
            "The most basic types of products (such as dress, sandals, etc) must be specified in the query in one word, "
            "and it should be the same as the item description. "
            "In addition, if the product specifies the gender, it must be added. "  # 衣服特有
            f"Also, the user's base information should be added to the query as a constraint if possible: {user}"
            "For example, gender and body type."  # 衣服特有
            "\n"
            # "Here are some examples:\n"
            # "Please recommend some suitable dresses to wear at a wedding\n"
            # "Is there any suitable trousers to wear when attending a meeting?\n"
            # # "I want some men T-shirts to go with gold necklaces\n"
            # "Please recommend some suitable men's jacket to wear to art exhibitions\n"
            # "Any recommendations for men's coat suitable for taking formal photos?\n"
            # # "Please recommend some dresses that would go well with purple heels.\n"
            # "I need some men's shoes for mountain biking.\n"
            # "Do you have any recommendations for skirts to wear to Disneyland?\n"
            # "I want some Women's Jackets that I can wear straight from a business meeting to a dinner party.\n"
            # "Please recommend some women's clothing for working in a museum\n"
            f"You must choose the most appropriate template from these to construct the query:{random_queries}"
            "If the template does not fit your needs, you can modify individual words to make the sentence natural and real."
            "Your query needs to be realistic because it is a real need. "
            "Do not include statements that may cause ambiguity. "
            # "No longer than 1 sentences."

            # " or 25 words"
        )
        if cat_number >= 10:
            prompt += "The total length should not exceed 2 sentences or 25 words"
        else:
            prompt += "The total length should not exceed 1 sentences"
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages, llm="gpt-4o-mini")
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
            "There are several ways to rewrite the query."
            "(1) If the requirement in the query can be met by a set of clothes, change the target piece of clothing to 'a set of clothes'.\n"
            "(2) If the query requirement can only be met by the specified clothing type, propose one or two more types of clothing to match the target clothing.\n"
            "(3) If the query is not particularly specific(For example, "
            "attending a dinner party is not a specific requirement, "
            "because it is reasonable to wear casual and comfortable clothes when dining with friends, "
            "but it is also reasonable to wear formal clothes when attending a company dinner.), "
            "add 'Different situations may need to be considered'. "
            f"The original query is {query}"
            f"The item description is {item}"
            f"The assigned number is {number}"
            "Please determine whether this query may conflict with the conditions of this method."
            "If there is no conflict, then still choose this number. Otherwise, output the number you think is most suitable."
            "You can be a little strict. Do not keep unreasonable classifications."
            "output the number in an '{}', '{}' contains pure number."
            "You must output a number!"
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages, llm="gpt-4o-mini")
        print(f"Chat Response: {response}")
        return response


    # def classify_query(self, query, item):
    #     # history = self.memory.get_history()
    #     # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
    #     # history_str = history
    #     sys_prompt = (
    #         "You are a troublesome shopper. You are good at rewriting queries in conversational recommendations to be more complex."
    #         # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
    #     )
    #     prompt = (
    #         "There are several ways to rewrite the query."
    #         "(1) If the requirement in the query can be met by a set of clothes, change the target piece of clothing to 'a set of clothes'.\n"
    #         "(2) If the query requirement can only be met by the specified clothing type, propose one or two more types of clothing to match the target clothing.\n"
    #         "(3) If the query is not particularly specific(For example, "
    #         "attending a dinner party is not a specific requirement, "
    #         "because it is reasonable to wear casual and comfortable clothes when dining with friends, "
    #         "but it is also reasonable to wear formal clothes when attending a company dinner.), "
    #         "add 'Different situations may need to be considered'. "
    #         "This method should be given a little higher priority. \n"
    #         f"The original query is {query}"
    #         f"The item description is {item}"
    #         "Please choose one of the three rewritings"
    #         "Please determine the most suitable rewriting method for this query and then output the number"
    #         "output the number in an '{}', '{}' contains pure number."
    #         "You must output a number!"
    #     )
    #     messages = [{"role": "system",
    #                  "content": sys_prompt},
    #                 {"role": "user",
    #                  "content": prompt}]
    #     response = get_completion(messages, llm="gpt-4o-mini")
    #     # print(f"Chat Response: {response}")
    #     return response

    # def classify_query_product(self, query, product):
    #     # history = self.memory.get_history()
    #     # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
    #     # history_str = history
    #     sys_prompt = (
    #         "You are a troublesome shopper. You are good at rewriting queries in conversational recommendations to be more complex."
    #         # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
    #     )
    #     prompt = (
    #         "There are several ways to rewrite the query."
    #         "(1) If the requirement in the query can be met by a set of product types, change the target piece of product to 'a set of products'.\n"
    #         # "(1) If Even if the specific product information is removed, there is still a demand for use, "
    #         # "then change the target piece of product to 'a set of products'.\n"
    #         "For example, 'Could you please recommend a #6 fishing hook without a barb? I'm looking for something specific for my fishing needs.'. "
    #         "can be transformed to 'Could you please recommend a set of products for my fishing needs.'. "
    #         # "(classification(1) needs to be given more consideration, because there are complementary products to use)\n"
    #         "(2) If the query requirement should be met by the specified product type, propose one or two more types of product to match the target clothing.\n"
    #
    #         f"The original query is {query}"
    #         f"The item description is {product}"
    #         "Please choose one of the two rewritings"
    #         "Please determine the most suitable rewriting method for this query and then output the number"
    #         "output the number in an '{}', '{}' contains **PURE** number without any character."
    #         "You **must** output a number!"
    #     )
    #     messages = [{"role": "system",
    #                  "content": sys_prompt},
    #                 {"role": "user",
    #                  "content": prompt}]
    #     response = get_completion(messages, llm="gpt-4o-mini")
    #     # print(f"Chat Response: {response}")
    #     return response

    def complement_query(self, query):
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = (
            "You are a conversational recommendation user and you are good at expressing your shopping needs in words."
            # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
        )
        prompt = (
            f"The key words describing your needs are: {query}."
            "This keyword may describe a product, or it may only describe a certain attribute of the target product."
            "Please rewrite this into a sentence expressing a request for a recommendation, such as: Please recommend me an xxx. "
            "The form of this sentence can be more free and not too rigid."
            "You need to transform the description of the item into a statement of actual usage requirements as much as possible."
            "Please pay special attention to keywords that begin with numbers or a combination of numbers and special characters(#,$,%,etc). "
            "They are not descriptions of the quantity to be purchased (they may be parameters such as model, size, etc.)."
            "You should keep these special characters + number combinations. "
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
            "You are a shopper. You are good at rewriting queries in conversational recommendations."
            # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
        )
        prompt = (
            f"The original query is {query}"
            # f"The item description is {item}"
        )
        if number == '1':
            prompt = prompt + (
                "The requirement in the query can be met by a set of clothes. "
                "Please change the target piece of clothing to 'a set of clothes'. "
                "For example, replace 'a dress' with 'a set of clothes'. "
                "'a set' is an important mark. "
                "Additionally, keep the description of men's, women's, boy's or girl's. "
            )
        elif number == '2':
            similar_items = self.get_top_n_similar_queries(query, df)
            concatenated_texts = ''
            for index, row in similar_items.iterrows():
                title = row['title']
                categories = row['category']
                concatenated_text = title + categories + '\n'  # 你可以根据需要调整拼接格式
                concatenated_texts += concatenated_text
            prompt = prompt + (
                "The query requirement can only be met by the specified clothing type, "
                "Add a sentence at the end to increase the matching of the target product."
                "For example: 'Additionally recommend an additional pair of pants and socks to go with these shoes'"
                f"The base types of target product should be chosen from following items:\n{concatenated_texts}."
            )

        elif number == '3':
            prompt = prompt + (
                "The query should be not particularly specific, "
                "Please change the query requirement to be a little less detailed and "
                "add this sentence at the end: ' I'm not sure about the specific wearing scene '. "
            )
        prompt = prompt + (
            "You cannot make major changes to the original query, such as adding new demand scenarios or "
            "replacing existing demand scenarios or add descriptive words for the target product. "
            "You can ONLY make appropriate additions or modifications to the query **STRICTLY** according to the instructions above!"
            "Please remember that the rewritten query should still be asked in the user's tone of voice just like the original one."
            "You are prohibited from outputting empty content."
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

    def rewrite_query_product(self, query, number, c_product_info):
        # history = self.memory.get_history()
        # history_str = "\n".join([f"User: {item['user']}\nAgent: {item['response']}" for item in history])
        # history_str = history
        sys_prompt = (
            "You are a shopper. You are good at rewriting queries in conversational recommendations."
            # "You are a shopper. You are good at asking queries that conversational recommendation systems cannot easily retrieve."
        )
        prompt = (
            f"The original query is {query}"
            f"First, you should identify what kind of product the user wants and what the requirements are for that product."
            # f"The item description is {item}"
        )

        # print(concatenated_texts)
        if number == '1':
            prompt = prompt + (
                "The requirement in the query can be met by a set of products. "
                f"The following are products that can be used together:\n{c_product_info}.\n"
                f"Please consider what this item can do with its complementary items, then propose such a demand for a set of products to be recommended. "
                # "And delete the product type originally required."
                "For example, 'Could you please recommend a #6 fishing hook without a barb? I'm looking for something specific for my fishing needs.'. "
                "This query with some other fishing tools"
                "can be transformed to 'Could you please recommend a set of products for my fishing needs. I'm specifically looking for a #6 fishing hook without a barb.'. "
                # "It is enough to change it to this one sentence."
                # "You **mustn't** add the item information in the original query to the new query."
                # "Additionally, keep the description of men's, women's, boy's or girl's. "
            )
        elif number == '2':
            prompt = prompt + (
                "The query requirement can only be met by the specified product type, "
                "Add a sentence at the end to increase the matching of the target product."
                "For example: 'Additionally recommend an additional pair of pants and socks to go with these shoes'"
                f"The following are complementary products that can be used:\n{c_product_info}.\n"
                "For complementary products, you can only mention the type of product and **mustn't** include any detailed description of the product attributes."
                "The description of the complementary product should not be more detailed than the original query."
                ""
            )
        prompt = prompt + (
            "You cannot make major changes to the original query, such as adding new demand scenarios or "
            "replacing existing demand scenarios or add descriptive words for the target product. "
            "But you should make the query more varied in terms of language style."
            "You can ONLY make appropriate additions or modifications to the query **STRICTLY** according to the instructions above!"
            "Please remember that the rewritten query should still be asked in the user's tone of voice."
        )
        messages = [{"role": "system",
                     "content": sys_prompt},
                    {"role": "user",
                     "content": prompt}]
        response = get_completion(messages, llm="gpt-4o-mini")
        print(number)
        print(prompt)
        return response

    def count_rec_number(self, query):
        sys_prompt = "You are a conversational recommendation user query understanding assistant."
        prompt = (
            f"User query is: \"{query}\"\n"
            "Please count how many types of products the user has requested in this query. "
            "Output a json format to include the quantity and specific products types of the required products."
            "For example: 'I'm looking for some merchandising tags without string. I need to identify the right type of product and its requirements. "
            "Additionally, could you recommend a tagging gun and some unstrung tags to go with them?' "
            "This query raises 3 requirements: merchandising tags without string, tagging gun, and unstrung tags."
            "However, 'Could you guide me to some women's clothes that are ideal for long city walks and casual outings with friends?'"
            "This is one product with multiple functions, it's still one type!"
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
            "You only need to output the revised query, with nothing else!"
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        response = get_completion(messages)
        return response.strip()

    def execute_task(self, task):
        return


