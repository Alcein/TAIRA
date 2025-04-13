# utils/task.py
import re

import yaml
from openai import OpenAI

with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
model = config['MODEL']



def get_completion(messages, llm=model, temperature=0):  # claude-3-5-sonnet-20240620 gpt-4o-2024-08-06 qwen-plus
    client = OpenAI(
        base_url="",
        api_key=""
    )
    tokens = 5000
    response = client.chat.completions.create(
        model=llm,
        messages=messages,
        temperature=1,
        timeout=50,
        max_tokens=tokens,
        top_p=0.1
    )
    # print("当前prompt：", messages)
    # print("当前llm输出：", response)

    return response.choices[0].message.content


def get_json(messages, json_format, llm=model, temperature=0):

    client = OpenAI(
        # base_url="",
        # api_key=""
    )

    response = client.beta.chat.completions.parse(
        model=llm,
        messages=messages,
        temperature=temperature,
        timeout=50,
        max_tokens=1000,
        response_format=json_format
    )

    return response.choices[0].message.content


def extract_braces_content(s):
    s = s.replace("\\'", "'")
    # 使用正则表达式匹配最前面的{和最后面的}之间的所有内容
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None
