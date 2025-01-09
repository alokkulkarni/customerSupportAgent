# type: ignore

import re
import requests
import numpy as np
import ollama
from openai import OpenAI
from langchain_core.tools import tool

response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        # self.api_key = api_key
        self.base_url = base_url

    def embeddings_create(self, model: str, input: list):
        url = f"{self.base_url}/api/embeddings"
        headers = {
            # "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"model": model, "input": input}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client=None):

        embeddings = client.embeddings.create(
            model="mxbai-embed-large",
            input=[doc["page_content"] for doc in docs],
        )
        vectors = [emb.embedding for emb in embeddings.data]

        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="mxbai-embed-large", prompt=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


retriever = VectorStoreRetriever.from_docs(docs, client)


@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events."""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])
