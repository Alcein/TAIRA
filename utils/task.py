# utils/task.py
import re

import yaml
from openai import OpenAI

with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
model = config['MODEL']



def get_completion(messages, llm=model, temperature=0):  # claude-3-5-sonnet-20240620 gpt-4o-2024-08-06 qwen-plus
    if 'qwen' in llm:
        client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-1f0acf1225cb4769b66f440394d14250"
        )
    else:
        client = OpenAI(
            # base_url="https://api.starteam.wang/v1",
            base_url="https://api.chatanywhere.tech/v1",
            # api_key="sk-lgO5N1o5LkB8kZ12Fc9071B9B2C0429cBdAe35Ae53351c66"
            api_key="sk-MW7MscGqMlYiMP0vT0kMEYl2jRouOWjwwUyCXe7NBEeXBke4"
        )
    if llm == 'qwen-max':
        tokens = 2000
    else:
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
        base_url="https://api.starteam.wang/v1",
        # base_url="https://api.chatanywhere.tech/v1",
        api_key="sk-lgO5N1o5LkB8kZ12Fc9071B9B2C0429cBdAe35Ae53351c66"
        # api_key="sk-MW7MscGqMlYiMP0vT0kMEYl2jRouOWjwwUyCXe7NBEeXBke4"
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
