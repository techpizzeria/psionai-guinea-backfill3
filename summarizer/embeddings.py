"""Embedding helpers built on the OpenAI Embedding API."""

import os

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-ada-002"


def embed_text(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Return the embedding vector for ``text``.

    Reads the API key from the ``OPENAI_API_KEY`` environment variable, which
    the caller is expected to have already loaded.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
