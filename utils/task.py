import re

import yaml
from openai import OpenAI

with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
model = config['MODEL']

def get_completion(messages, llm=model, temperature=0):
    client = OpenAI(
        base_url=config["OPENAI_BASE_URL"],
        api_key=config["OPENAI_API_KEY"]
    )
    response = client.chat.completions.create(
        model=llm,
        messages=messages,
        temperature=1,
        timeout=50,
        max_tokens=5000,
        top_p=0.1
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
