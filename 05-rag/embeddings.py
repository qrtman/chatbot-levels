"""Generate Gemini embeddings for documents and search queries."""

import math
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-001"


class GeminiEmbedder:
    """Convert text into numerical vectors using Gemini."""

    def __init__(self, client=None, model=EMBEDDING_MODEL):
        self.model = model

        if client is None:
            root = Path(__file__).resolve().parent.parent
            load_dotenv(root / ".env")
            client = genai.Client()

        self.client = client

    def _embed_texts(self, texts, task_type):
        """Generate and validate embedding vectors."""

        if not isinstance(texts, list):
            raise TypeError("texts must be a list")

        if not texts:
            return []

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Embedding inputs must be nonempty strings"
            )

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
            ),
        )

        embeddings = response.embeddings

        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding response count does not match input count"
            )

        vectors = []

        for embedding in embeddings:
            values = embedding.values

            if not values:
                raise RuntimeError("Received an empty embedding")

            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(
                    "Received non-finite embedding values"
                )

            vectors.append(list(values))

        dimension = len(vectors[0])

        if any(len(vector) != dimension for vector in vectors):
            raise RuntimeError(
                "Embedding dimensions are inconsistent"
            )

        return vectors

    def embed_documents(self, texts):
        """Embed document chunks for indexing."""

        return self._embed_texts(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
        )

    def embed_query(self, question):
        """Embed a user question for semantic retrieval."""

        vectors = self._embed_texts(
            [question],
            task_type="RETRIEVAL_QUERY",
        )

        return vectors[0]