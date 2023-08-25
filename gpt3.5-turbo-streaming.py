import openai
from dotenv import dotenv_values

config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

for data in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a McKinsey consultant"},
            {"role": "user", "content": "Please write a summary about McKinsey & Company no more than 200 words"}
        ],
        stream=True,
):
    print(data.choices[0].delta.content, end="", flush=True)
