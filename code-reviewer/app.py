import openai
from dotenv import load_dotenv
import os
import argparse

prompt = """
    You will receive a file's content as text. Generate a code review for the file in short language. Indicate what 
    changes should be made to improve its style, performance, readability and maintainability. If there are any 
    reputable libraries that could be introduced to improve the code, suggest them. Be kind and constructive. 
    For each suggested change, include line numbers to which you are referring.
"""

content_old = """
    def get_message(input_msg):
        messages.append({"role": "user", "content": input_msg})
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200
        )
        message = resp["choices"][0].message
        messages.append(message.to_dict())
        return message.content
"""


def code_review(file_path: str, model: str) -> None:
    with open(file_path, "r") as file:
        content = file.read()
        make_code_review_request(content, model)


def make_code_review_request(content: str, model: str) -> None:
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Code review for the following file: {content}"}
    ]

    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    print(resp.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(description="Simple code reviewer for single file")
    parser.add_argument("f")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    args = parser.parse_args()
    code_review(args.f, args.model)


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
