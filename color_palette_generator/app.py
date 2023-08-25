import openai
from dotenv import dotenv_values
from IPython.display import Markdown, display
import json
from flask import Flask, render_template, request

app = Flask(__name__,
            template_folder="../templates",
            static_url_path="",
            static_folder="../static/color_palette_generator")
config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/palette", methods=["POST"])
def prompt_to_palette():
    query = request.json['query']
    app.logger.info(f"Request received, {query}")
    colors = get_color2(query)
    return {"colors": colors}


def get_colors(msg):
    prompt = f"""
        You are a color palette generating assistant that responds to text prompts for color palettes
        You should generate color palettes that fit the theme, mood, or instructions in the prompt.
        The palettes should be between 2 to 8 colors.

        Q: Convert the following verbal description of a color palette into a list of colors: The Mediterranean Sea
        A: ["#006699", "#66CCCC", "#F0E68C", "#008000", "#F08080"]

        Q: Convert the following verbal description of a color palette into a list of colors: sage, nature, earth
        A: ["#EDF1D6", "#9DC08B", "#609966", "#40513B"]

        Desired format: a JSON array of hexadecimal color codes
        Q:Convert the following verbal description of a color palette into a list of colors: {msg}
        A:
    """
    response = openai.Completion.create(
        prompt=prompt,
        model="text-davinci-003",
        max_tokens=200,
        stop="11."
    )

    print(response.choices[0].text.strip())
    colors = json.loads(response.choices[0].text.strip())
    return colors


def get_color2(msg):
    messages = [
        {"role": "system", "content": """You are a color palette generating assistant that responds to text prompts for 
                                            color palettes You should generate color palettes that fit the theme, mood, 
                                            or instructions in the prompt."""},
        {"role": "user", "content": f"""Convert the following verbal description of a color 
                                            palette into a list of colors: The Mediterranean Sea"""},
        {"role": "assistant", "content": '["#006699", "#66CCCC", "#F0E68C", "#008000", "#F08080"]'},
        {"role": "user", "content": f""" Convert the following verbal description of a color 
                                             palette into a list of colors: sage, nature, earth"""},
        {"role": "assistant", "content": '["#EDF1D6", "#9DC08B", "#609966", "#40513B"]'},
        {"role": "user", "content": f"""Convert the following verbal description of a color 
                                             palette into a list of colors: {msg}"""},
    ]
    response = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=200
    )
    print(response)
    colors = json.loads(response.choices[0].message.content.strip())
    return colors


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
