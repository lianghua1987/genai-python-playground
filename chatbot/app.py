import openai
from dotenv import dotenv_values
import argparse
import enum

config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

parser = argparse.ArgumentParser(description="Simple command line chatbot with GPT-3.5-turbo")
parser.add_argument("--personality", type=str, help="A brief summary of chatbot personality",
                    default="friendly and helpful chatbot")

args = parser.parse_args()
personality = args.personality
print(f"Personality set to: {personality}")

messages = [{"role": "system", "content": f"""You are a conversational chatbot. Your personality is {personality}"""}]


class Style(enum.Enum):
    bold = "\033[1m"
    red = "\033[31m"
    blue = "\033[34m"


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


def color(text, style):
    return f"{style}{text}\033[0m"


def main():
    while True:
        try:
            user_input = input(color(color("You:", Style.blue.value), Style.bold.value))
            result = get_message(user_input)
            print(f"{color(color('Assistant: ', Style.red.value), Style.bold.value)} {result}")
        except KeyboardInterrupt:
            print("Exiting...")
            break


if __name__ == '__main__':
    main()
