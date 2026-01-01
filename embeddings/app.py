import numpy as np
import openai
import pandas as pd
import pickle
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import dotenv_values
import tiktoken
import time
from nomic import atlas
from openai.embeddings_utils import distances_from_embeddings, indices_of_nearest_neighbors_from_distances

from utils import Bgcolors as colors

default_model = "text-embedding-ada-002"
embedding_cache_path = "movie_embeddings_cache.pkl"


@retry(wait=wait_random_exponential(min=40, max=60), stop=stop_after_attempt(3))
def get_embedding(text: str, model=default_model):
    response = openai.Embedding.create(
        input="candy canes",
        model=model
    )
    return response["data"][0]["embedding"]


# Establish a cache of embeddings to avoid recomputing
# Cache os a doct pf tuples(text, model) -> embedding, saved as a pickle file

# Set path to embedding cache

def embedding_from_string(string, model, embedding_cache):
    """Return embedding of given string, using a cache to avoid recomputing..."""
    if (string, model) not in embedding_cache.keys():
        start_time = time.time()
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"{colors.OKGREEN}INFO: {colors.ENDC}Got embedding for - {string[:28]}, "
              f"Execution time: {colors.OKCYAN}{'{:.2f}'.format(time.time() - start_time)}{colors.ENDC}s")
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


def cal_tokens(enc, string: str):
    return len(enc.encode(string))


def check_tokens(plots):
    enc = tiktoken.encoding_for_model(default_model)
    return sum([cal_tokens(enc, plot) for plot in plots])


def cal_cost(num_of_tokens: int) -> float:
    return .0004 / 1000 * num_of_tokens


def map_embeddings(plot_embeddings, data):
    atlas.map_embeddings(
        embeddings=np.array(plot_embeddings),
        data=data
    )


def get_recommendations(plots, index_of_sources_string, embedding_cache, k_nearest_neighbors=5, model=default_model):
    # Get all the embeddings
    embeddings = [embedding_from_string(plot, default_model, embedding_cache) for plot in plots]

    # Get embedding for specific query string
    query_embedding = embeddings[index_of_sources_string]

    # Get distance between our embedding and all other embedding
    distance = distances_from_embeddings(query_embedding, embeddings)

    # Get indices of the nearest neighbors
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distance)
    plot = plots[index_of_sources_string]
    count = 0
    for i in indices_of_nearest_neighbors:
        if plot == plots[i]:
            continue
        if count >= k_nearest_neighbors:
            break
        count += 1
        print(f"Found {count} closet match with distances of {colors.OKBLUE}{distance[i]}{colors.ENDC}")


def main():
    data_sample_path = "movie_plots.csv"
    df = pd.read_csv(data_sample_path)

    # Load cache if exists, and save a copy to disk
    try:
        embedding_cache = pd.read_pickle(embedding_cache_path)
    except FileNotFoundError:
        print(f"{colors.WARNING}WARNING: {colors.ENDC}File does not found, new one created")
        embedding_cache = {}
    except EOFError as e:
        print(f"{colors.FAIL}ERROR: {colors.ENDC}{e}")

    with open(embedding_cache_path, "wb") as embedding_cache_file:
        pickle.dump(embedding_cache, embedding_cache_file)

    movies = df[df["Origin/Ethnicity"] == "American"].sort_values("Release Year", ascending=False).head(5000)
    print(movies)
    plots = movies["Plot"].values
    tokens = check_tokens(plots)
    estimated_cost = cal_cost(tokens)
    print(
        f"Estimated cost: {colors.BOLD}${float('{:.2f}'.format(estimated_cost))}{colors.ENDC} with total {tokens} tokens.")

    plot_embeddings = [embedding_from_string(plot, default_model, embedding_cache) for plot in plots]
    data = movies[["Title", "Genre"]].to_dict("records")
    map_embeddings(plot_embeddings, data)

    get_recommendations(plots, 0, embedding_cache, 10)


if __name__ == '__main__':
    config = dotenv_values("../.env")
    openai.api_key = config["OPENAI_API_KEY"]
    main()
