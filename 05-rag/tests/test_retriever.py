"""Offline tests for semantic retrieval."""

import sys
import unittest

from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import chromadb

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from retriever import retrieve


class RetrieverTests(unittest.TestCase):

    def setUp(self):
        self.collection = Mock()
        self.embedder = Mock()

        self.collection.count.return_value = 2
        self.embedder.embed_query.return_value = [
            1.0, 0.0, 0.0
        ]

        self.collection.query.return_value = {
            "ids": [["shipping-1", "returns-1"]],
            "documents": [[
                "Shipping takes 3-5 business days.",
                "Returns are accepted within 30 days.",
            ]],
            "metadatas": [[
                {"source": "shipping.md", "chunk_index": 0},
                {"source": "returns.md", "chunk_index": 0},
            ]],
            "distances": [[0.1, 0.8]],
        }

    def test_returns_ranked_results(self):
        results = retrieve(
            "How long does shipping take?",
            self.collection,
            self.embedder,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["source"], "shipping.md")
        self.assertEqual(results[0]["distance"], 0.1)
        self.assertEqual(results[1]["source"], "returns.md")

    def test_uses_query_embedding(self):
        retrieve(
            "  Where is my package?  ",
            self.collection,
            self.embedder,
        )

        self.embedder.embed_query.assert_called_once_with(
            "Where is my package?"
        )

        kwargs = self.collection.query.call_args.kwargs

        self.assertEqual(
            kwargs["query_embeddings"],
            [[1.0, 0.0, 0.0]],
        )

    def test_limits_results_to_collection_size(self):
        retrieve(
            "Shipping question",
            self.collection,
            self.embedder,
            top_k=10,
        )

        kwargs = self.collection.query.call_args.kwargs

        self.assertEqual(kwargs["n_results"], 2)

    def test_empty_collection(self):
        self.collection.count.return_value = 0

        results = retrieve(
            "Shipping question",
            self.collection,
            self.embedder,
        )

        self.assertEqual(results, [])

        self.embedder.embed_query.assert_not_called()
        self.collection.query.assert_not_called()

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            retrieve(
                "   ",
                self.collection,
                self.embedder,
            )

        self.embedder.embed_query.assert_not_called()

    def test_rejects_non_string_question(self):
        with self.assertRaises(ValueError):
            retrieve(
                None,
                self.collection,
                self.embedder,
            )

    def test_rejects_invalid_top_k(self):
        for value in [0, -1, True, 1.5]:
            with self.subTest(top_k=value):
                with self.assertRaises(ValueError):
                    retrieve(
                        "Shipping",
                        self.collection,
                        self.embedder,
                        top_k=value,
                    )

    def test_rejects_invalid_chroma_response(self):
        self.collection.query.return_value = {
            "ids": [["shipping-1"]],
            "documents": None,
            "metadatas": [[{"source": "shipping.md"}]],
            "distances": [[0.1]],
        }

        with self.assertRaises(RuntimeError):
            retrieve(
                "Shipping",
                self.collection,
                self.embedder,
            )

    def test_rejects_missing_source(self):
        self.collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Some policy text"]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        }

        with self.assertRaises(RuntimeError):
            retrieve(
                "Shipping",
                self.collection,
                self.embedder,
            )

    def test_embedding_failure_propagates(self):
        self.embedder.embed_query.side_effect = RuntimeError(
            "Embedding API unavailable"
        )

        with self.assertRaises(RuntimeError):
            retrieve(
                "Shipping",
                self.collection,
                self.embedder,
            )

        self.collection.query.assert_not_called()

    def test_real_local_chroma_search(self):
        """Test similarity ranking without external APIs."""

        client = chromadb.EphemeralClient()

        collection = client.create_collection(
            name=f"retrieval_{uuid4().hex[:16]}",
            embedding_function=None,
            configuration={
                "hnsw": {"space": "cosine"}
            },
        )

        collection.add(
            ids=[
                "shipping-1",
                "returns-1",
                "refunds-1",
            ],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            documents=[
                "Shipping takes 3-5 days.",
                "Returns are allowed for 30 days.",
                "Refunds take 5-7 days.",
            ],
            metadatas=[
                {"source": "shipping.md", "chunk_index": 0},
                {"source": "returns.md", "chunk_index": 0},
                {"source": "refunds.md", "chunk_index": 0},
            ],
        )

        embedder = Mock()
        embedder.embed_query.return_value = [
            1.0, 0.0, 0.0
        ]

        results = retrieve(
            "Shipping question",
            collection,
            embedder,
            top_k=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "shipping-1")
        self.assertEqual(results[0]["source"], "shipping.md")
        self.assertAlmostEqual(
            results[0]["distance"],
            0.0,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()