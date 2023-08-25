import openai
from dotenv import dotenv_values

config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

prompt = """Tell me a funny but inspiration story no more than 200 words to get me motivated"""
for data in openai.Completion.create(
        prompt=prompt,
        model="text-davinci-003",
        max_tokens=200,
        stream=True
):
    print(data.choices[0].text, end="", flush=True)
