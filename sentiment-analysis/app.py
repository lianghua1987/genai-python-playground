import openai
from openai.error import APIConnectionError, APIError, RateLimitError
from typing import List, Dict, Generator, Optional

import tiktoken
import pandas as pd
from dotenv import dotenv_values
import sys

import re
import random
import time
import praw

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from tenacity import retry, wait_random_exponential, retry_if_exception_type, stop_after_attempt

config = dotenv_values("../.env")
openai.api_key = config["OPENAI_API_KEY"]

default_model = "gpt-3.5-turbo"
DF_COLUMNS = ["subreddit", "submission_id", "score", "comment_body"]
default_filename, subreddits = "cities.csv", ["Philadelphia", "Atlanta", "Orlando", "Arlington", "Shanghai"]


def get_reddit():
    return praw.Reddit(
        client_id=config["REDDIT_CLIENT_ID"],
        client_secret=config["REDDIT_CLIENT_SECRET"],
        user_agent=f"script:test:0.0.1 (by u/RowEnvironmental7282)"
    )


def comment_generator(submission) -> Generator:
    for comment in submission.comments.list():
        if hasattr(comment, "body") and comment.body != "[deleted]" and comment.body != "[removed]":
            yield comment


def collect_comments(filename: str, target_comments_per_subreddit: int, max_comments_per_submission: int,
                     max_comment_length: int, reddit: praw.Reddit) -> pd.DataFrame:
    """
    Collect comments from the top submissions in each subreddit. Cache results at cache_filename, return a dataframe
    with columns: subreddit, submission_id, score, comment_body
    """

    try:
        df = pd.read_csv(filename, index_col="id")
        assert df.columns.tolist() == DF_COLUMNS
    except FileNotFoundError:
        df = pd.DataFrame(columns=DF_COLUMNS)

    records = df.to_dict(orient="index")
    for subreddit_index, subreddit_name in enumerate(subreddits):
        print(f"Processing subreddit: {subreddit_name}")

        processed_comments_for_subreddit = len(df[df["subreddit"] == subreddit_name])

        if processed_comments_for_subreddit >= target_comments_per_subreddit:
            print(f"Enough comments fetched for {subreddit_name}, continuing to next subreddit.")
            continue

        for submission in reddit.subreddit(subreddit_name).top(time_filter="month"):
            if processed_comments_for_subreddit >= target_comments_per_subreddit:
                break

            processed_comments_for_submission = len(df[df["submission_id"] == submission.id])

            for comment in comment_generator(submission):
                if (processed_comments_for_submission >= max_comments_per_submission
                        or processed_comments_for_subreddit >= target_comments_per_subreddit):
                    break

                if comment.id in records:
                    print(
                        f"Skipping comment {subreddit_name}-{submission.id}-{comment.id} because we already have it")
                    continue

                body = comment.body[:max_comment_length].strip()
                records[comment.id] = {"subreddit": subreddit_name, "submission_id": submission.id,
                                       "comment_body": body}

                processed_comments_for_submission += 1
                processed_comments_for_subreddit += 1

        # Write to disk
        print(f"CSV rewritten with {len(records)} rows.\n")
        df = pd.DataFrame.from_dict(records, orient="index", columns=DF_COLUMNS)
        df.to_csv(filename, index_label="id")

    print("Completed")
    return df


def generate_prompt(s: str) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": """
    The following is a comment from a user on Reddit. Score it from -1 to 1, where -1 is the most negative and 1 is the most positive:

    The traffic is quite annoying.
    """.strip(),
        },
        {"role": "assistant", "content": "-0.75"},
        {
            "role": "user",
            "content": """
    The following is a comment from a user on Reddit. Score it from -1 to 1, where -1 is the most negative and 1 is the most positive:

    The library is downtown.
    """.strip(),
        },
        {"role": "assistant", "content": "0.0"},
        {
            "role": "user",
            "content": """
    The following is a comment from a user on Reddit. Score it from -1 to 1, where -1 is the most negative and 1 is the most positive:

    Even though it's humid, I really love the summertime. Everything is so green and the sun is out all the time.
    """.strip(),
        },
        {"role": "assistant", "content": "0.8"},
        {
            "role": "user",
            "content": f"""
    The following is a comment from a user on Reddit. Score it from -1 to 1, where -1 is the most negative and 1 is the most positive:

    {s}
    """.strip(),
        },
    ]


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def estimate_cost(df, model):
    tokens = 0
    for comment in df["comment_body"]:
        tokens += num_tokens_from_messages(generate_prompt(comment), model)
    const = tokens * 0.002 / 1000
    print(f"Scoring {len(df)} comments will cost approximately ${const:.2f}")


class UnscorableCommentError(Exception):
    pass


@retry(wait=wait_random_exponential(multiplier=1, max=5), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(UnscorableCommentError) | retry_if_exception_type(
           APIConnectionError) | retry_if_exception_type(RateLimitError), reraise=True)
def get_score(comment: str, model: str):
    messages = generate_prompt(comment)
    response = openai.ChatCompletion.create(
        model=default_model,
        messages=messages
    )
    try:
        return float(response.choices[0].message.content)
    except:
        print(f"Can't score for comment: {comment}")
        raise UnscorableCommentError(f"Can't score for comment: {comment}")


def score_sentiments(df, model: str):
    records = df.to_dict(orient="index")

    for index, item in enumerate(records.items()):
        comment_id, comment = item

        if not pd.isna(comment["score"]):
            print(f"{comment_id} already scored, skipping.")
            continue

        body = comment["comment_body"]
        try:
            score = get_score(body, model)
        except UnscorableCommentError:
            continue
        print(
            f"""
            {comment_id} - ({index + 1} of {len(records)} Comments)
            Body: {body[:80]}
            Score: {score}""".strip()
        )
        records[comment_id]["score"] = score
        df = pd.DataFrame.from_dict(records, orient="index", columns=DF_COLUMNS)
        df.to_csv(default_filename, index_label="id")


def get_avg_score_by_subreddit(dataframe):
    """
    Given a pandas DataFrame with columns "subreddit" and "score", returns a new DataFrame
    with the average score and standard deviation for each subreddit.
    """
    # Group by subreddit and calculate the mean and standard deviation for each group
    subreddit_stats = dataframe.groupby("subreddit")["score"].agg(["mean", "std"])

    # Rename columns to indicate that they represent the mean and standard deviation
    subreddit_stats.columns = ["mean_score", "standard_deviation"]

    subreddit_stats = subreddit_stats.sort_values("mean_score", ascending=True)

    # Return the new DataFrame
    return subreddit_stats


def plot_sentiments(df):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Create the data
    df_scores = df[["score", "subreddit"]]

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
    g = sns.FacetGrid(df_scores, row="subreddit", row_order=get_avg_score_by_subreddit(df_scores).index.to_list(),
                      hue="subreddit", aspect=15, height=0.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "score", bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "score", clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "score")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    display(g.fig)
    plt.show()


def main():
    print("Start semantic analyzing")
    reddit = get_reddit()
    collect_comments(
        filename=default_filename,
        target_comments_per_subreddit=50,
        max_comments_per_submission=10,
        max_comment_length=1000,
        reddit=reddit
    )
    df = pd.read_csv(default_filename, index_col="id")
    assert df.columns.tolist() == DF_COLUMNS

    estimate_cost(df, default_model)
    score_sentiments(df, default_model)
    plot_sentiments(df)

if __name__ == "__main__":
    main()
