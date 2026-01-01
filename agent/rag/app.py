from typing import List

import sys, platform, os


from sentence_transformers import SentenceTransformer


def split_into_chunks(file: str) -> List[str]:
    with open(file, 'r') as file:
        content = file.read()
    return [chunk for chunk in content.split("\n\n")]


# embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")



def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk)
    return embedding.tolist()


chunks = split_into_chunks("resume.txt")
# for i, chunk in enumerate(chunks):
#     print(f"[{i}] {chunk}\n")

# test_embedding = embed_chunk("测试梁骅的简历")
# print(len(test_embedding))
# print(test_embedding)
#
embeddings = [embed_chunk(chunk) for chunk in chunks]
# print(len(embeddings))
# print(embeddings)

import chromadb

chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")


def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    ids = [str(i) for i in range(len(chunks))]
    chromadb_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )


save_embeddings(chunks, embeddings)


def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results['documents'][0]


query = "Does this guy have experience using Java? If so, for how many years? "
retrieve_chunks = retrieve(query, 20)

for i, chunk in enumerate(retrieve_chunks):
    print(f"[{i}] {chunk}\n")

from sentence_transformers import CrossEncoder


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)
    print(f"scores: {scores}")

    chunk_with_score_list = [(chunk, score) for chunk, score in zip(retrieved_chunks, scores)]
    chunk_with_score_list.sort(key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in chunk_with_score_list]


reranked_chunks = rerank(query, retrieve_chunks, 5)

for i, chunk in enumerate(reranked_chunks):
    print(f"[{i}] {chunk}\n")

from dotenv import load_dotenv
from google import genai

load_dotenv()
google_client = genai.Client()


def generate(query: str, chunks: List[str]) -> str:
    prompt = f"""
        You are a tech company recruiter, generate the answer based on the the questions and below information from resume.
        
        user question: {query}
        
        reference from resume:
        {"$n$n".join(chunks)}
        
        Make sure generate the answer based on the information provided, DO NOT MAKE UP THINGS.
""".replace("$n", "\n")

    print(f"prompt: {prompt}")

    response = google_client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt
    )

    return response.text

answer = generate(query, reranked_chunks)
print(answer)


