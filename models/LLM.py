from config import Config
from openai import OpenAI


def get_completion_deepseek(user_input, model="deepseek-chat"):
    deepseek_api_key = Config.DEEPSEEK_API_KEY
    base_url = Config.DEEPSEEK_BASE_URL

    messages = [{"role": "user", "content": user_input}]
    client = OpenAI(api_key=deepseek_api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content


# def get_completion_openrouter(user_input, model=model_name):
#     messages = [{"role": "user", "content": user_input}]
#     client = OpenAI(api_key=api_key, base_url=base_url)
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0,  # 模型输出的随机性，0 表示随机性最小
#         n=10
#     )
#     return response.choices[0].message.content

def get_intent_deepseek(user_input, model="deepseek-chat"):
    deepseek_api_key = Config.DEEPSEEK_API_KEY
    base_url = Config.DEEPSEEK_BASE_URL

    if isinstance(user_input, dict):
        content = user_input.get('user_query', '')
    else:
        content = str(user_input)

    messages = [{"role": "user", "content": content}]
    client = OpenAI(api_key=deepseek_api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content