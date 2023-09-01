import openai
from dotenv import dotenv_values

config = dotenv_values("../.env")
openai.api_key = config["OPENAI_API_KEY"]
def wisper(file):
    resp = openai.Audio.transcribe(
        model="whisper-1",
        file=file
    )
    return resp.text


def main():
    audio_file = open("./audio/elon.mp3", "rb")
    text = wisper(audio_file)
    print(text)


if __name__ == '__main__':
    main()
