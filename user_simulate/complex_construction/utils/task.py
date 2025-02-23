# utils/task.py

from openai import OpenAI


def get_completion(messages, llm="gpt-4o-2024-08-06", json_format=None, temperature=0):
    client = OpenAI(
        # base_url="https://api.starteam.wang/v1",
        base_url="https://api.chatanywhere.tech/v1",
        # api_key="sk-lgO5N1o5LkB8kZ12Fc9071B9B2C0429cBdAe35Ae53351c66"
        api_key="sk-MW7MscGqMlYiMP0vT0kMEYl2jRouOWjwwUyCXe7NBEeXBke4"
    )
    # response = client.chat.completions.create(
    #     model=llm,
    #     messages=messages,
    #     temperature=temperature,
    #     timeout=50,
    #     max_tokens=1000
    # )
    if json_format is None:
        response = client.chat.completions.create(
            model=llm,
            messages=messages,
            temperature=temperature,
            timeout=50,
            max_tokens=1000
        )
    else:
        response = client.beta.chat.completions.parse(
            model=llm,
            messages=messages,
            temperature=temperature,
            timeout=50,
            max_tokens=1000,
            response_format=json_format
        )

    return response.choices[0].message.content


def main():
    messages = [{"role": "system",
                 "content": "You are a shopper."},
                {"role": "user",
                 "content": "Hello."}]
    response = get_completion(messages, llm='gpt-4o')
    print(response)


if __name__ == "__main__":
    main()
