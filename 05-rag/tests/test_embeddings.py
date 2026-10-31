"""Offline tests for Gemini embedding integration."""

import sys
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings import GeminiEmbedder


class EmbeddingTests(unittest.TestCase):

    def setUp(self):
        self.client = Mock()
        self.embedder = GeminiEmbedder(client=self.client)

    def mock_embeddings(self, vectors):
        self.client.models.embed_content.return_value = (
            SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=vector)
                    for vector in vectors
                ]
            )
        )

    def task_type_used(self):
        config = (
            self.client.models.embed_content.call_args
            .kwargs["config"]
        )

        task_type = config.task_type

        return getattr(task_type, "value", task_type)

    def test_document_embeddings(self):
        self.mock_embeddings([
            [0.1, 0.2],
            [0.3, 0.4],
        ])

        result = self.embedder.embed_documents([
            "Shipping takes 3 days",
            "Returns take 30 days",
        ])

        self.assertEqual(result, [
            [0.1, 0.2],
            [0.3, 0.4],
        ])

        self.assertEqual(
            self.task_type_used(),
            "RETRIEVAL_DOCUMENT",
        )

    def test_query_embedding(self):
        self.mock_embeddings([[0.5, 0.6]])

        result = self.embedder.embed_query(
            "How long does shipping take?"
        )

        self.assertEqual(result, [0.5, 0.6])

        self.assertEqual(
            self.task_type_used(),
            "RETRIEVAL_QUERY",
        )

    def test_correct_model_used(self):
        self.mock_embeddings([[0.1, 0.2]])

        self.embedder.embed_query("Test question")

        self.client.models.embed_content.assert_called_once()

        kwargs = (
            self.client.models.embed_content.call_args.kwargs
        )

        self.assertEqual(
            kwargs["model"],
            "gemini-embedding-001",
        )

    def test_empty_document_list(self):
        result = self.embedder.embed_documents([])

        self.assertEqual(result, [])

        self.client.models.embed_content.assert_not_called()

    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            self.embedder.embed_query("   ")

        self.client.models.embed_content.assert_not_called()

    def test_invalid_document_rejected(self):
        with self.assertRaises(ValueError):
            self.embedder.embed_documents(["Valid", ""])

    def test_missing_embeddings_rejected(self):
        self.client.models.embed_content.return_value = (
            SimpleNamespace(embeddings=[])
        )

        with self.assertRaises(RuntimeError):
            self.embedder.embed_query("Test")

    def test_embedding_count_mismatch(self):
        self.mock_embeddings([[0.1, 0.2]])

        with self.assertRaises(RuntimeError):
            self.embedder.embed_documents([
                "Document one",
                "Document two",
            ])

    def test_inconsistent_dimensions(self):
        self.mock_embeddings([
            [0.1, 0.2],
            [0.3, 0.4, 0.5],
        ])

        with self.assertRaises(RuntimeError):
            self.embedder.embed_documents([
                "Document one",
                "Document two",
            ])

    def test_nonfinite_values_rejected(self):
        self.mock_embeddings([
            [float("nan"), 0.2]
        ])

        with self.assertRaises(RuntimeError):
            self.embedder.embed_query("Test")


if __name__ == "__main__":
    unittest.main()