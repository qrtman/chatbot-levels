"""Retrieve relevant knowledge-base chunks using ChromaDB."""

import argparse
import math

from embeddings import GeminiEmbedder
from ingest import open_collection


DEFAULT_TOP_K = 3


def retrieve(
    question: str,
    collection,
    embedder,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Retrieve the most relevant chunks for a question."""

    # Validate input.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string")

    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    # Avoid embedding requests when the database is empty.
    document_count = collection.count()

    if document_count == 0:
        return []

    # Generate the query embedding.
    query_vector = embedder.embed_query(question.strip())

    # Search the existing ChromaDB collection.
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, document_count),
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns nested lists, one per input query.
    fields = [
        results.get("ids"),
        results.get("documents"),
        results.get("metadatas"),
        results.get("distances"),
    ]

    if any(
        not isinstance(field, list) or len(field) != 1
        for field in fields
    ):
        raise RuntimeError("Invalid ChromaDB query response")

    ids, documents, metadatas, distances = [
        field[0] for field in fields
    ]

    if len(set(map(len, [
        ids,
        documents,
        metadatas,
        distances,
    ]))) != 1:
        raise RuntimeError(
            "ChromaDB returned inconsistent result lengths"
        )

    retrieved = []

    for chunk_id, text, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        if not isinstance(metadata, dict):
            raise RuntimeError("Missing chunk metadata")

        source = metadata.get("source")

        if not isinstance(source, str) or not source:
            raise RuntimeError("Missing source attribution")

        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Invalid retrieved document")

        if not math.isfinite(distance):
            raise RuntimeError("Invalid similarity distance")

        retrieved.append({
            "id": chunk_id,
            "text": text,
            "source": source,
            "chunk_index": metadata.get("chunk_index"),
            "distance": float(distance),
            "metadata": metadata,
        })

    return retrieved


def main():
    """Run a single semantic search from the command line."""

    parser = argparse.ArgumentParser(
        description="Search the Demo Store knowledge base"
    )

    parser.add_argument(
        "question",
        help="Natural-language question to search for",
    )

    args = parser.parse_args()

    collection = open_collection()

    if collection.count() == 0:
        parser.exit(
            1,
            "Knowledge base is empty. "
            "Run python 05-rag/ingest.py first.\n",
        )

    embedder = GeminiEmbedder()

    results = retrieve(
        question=args.question,
        collection=collection,
        embedder=embedder,
    )

    if not results:
        print("No documents found.")
        return

    print(f"\nFound {len(results)} relevant chunks:\n")

    for index, result in enumerate(results, start=1):
        print(f"Result {index}")
        print(f"Source: {result['source']}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text:\n{result['text']}")
        print("-" * 50)


if __name__ == "__main__":
    main()