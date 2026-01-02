import argparse
import enum
import os
import requests
import openai
from dotenv import dotenv_values
from IPython.display import Image
import base64

config = dotenv_values("../.env")
openai.api_key = config["OPENAI_API_KEY"]


def save_images(data, filename: str, format: str):
    image_dir_name = "images"
    image_directory = os.path.join(os.curdir, image_dir_name)

    if not os.path.isdir(image_directory):
        os.makedirs(image_directory)

    image_file_path = os.path.join(image_directory, filename)
    print(f"Image path: {image_file_path}")

    with open(image_file_path, "wb") as image_file:
        if not format == "b64_json":
            print(f"Image url: {data}")
            image = requests.get(data).content
            image_file.write(image)
        else:
            image_file.write(base64.b64decode(data))


def get_image(prompt, response_format="url"):
    res = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format=response_format
    )
    if response_format == "url":
        return res["data"][0]["url"]
    else:
        return res["data"][0]["b64_json"]


def get_variation(base_image_path: str):
    res = openai.Image.create_variation(
        image=open(base_image_path, "rb"),
        n=1
    )
    return res["data"][0]["url"]

def main():
    # data = get("A cute cat lounging in a tropical resort in Caribbean sea, digit art", "url")
    # data = get_image("A cute cat lounging in a tropical resort in Caribbean sea, digit art", "b64_json")
    # save_images(data, "download.png", "b64_json")

    data = get_variation("./images/cat.png")
    save_images(data, "variation.png", "url")



if __name__ == '__main__':
    main()
