"""Offline integration tests for the complete RAG pipeline."""

import json
import sys
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import chromadb

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from bot import (
    CLARIFICATION_ANSWER,
    RAGChatbot,
    REWRITE_PROMPT,
    UNSUPPORTED_ANSWER,
)
from ingest import prepare_chunks, sync_collection
from retriever import retrieve


# Small, deterministic knowledge base for integration testing.
POLICIES = {
    "shipping.md": (
        "# Shipping Policy\n"
        "Standard shipping takes 3-5 business days after dispatch."
    ),
    "returns.md": (
        "# Returns Policy\n"
        "Eligible returns are accepted within 30 calendar days."
    ),
    "refunds.md": (
        "# Refunds Policy\n"
        "Approved refunds are processed in 5-7 business days."
    ),
    "tracking.md": (
        "# Tracking Policy\n"
        "Track your order from your store account."
    ),
    "support.md": (
        "# Support Policy\n"
        "Contact support at support@demostore.example."
    ),
}


def fake_vector(text):
    """Create deterministic test vectors without Gemini."""

    text = text.lower()

    if "track" in text:
        index = 3
    elif "refund" in text:
        index = 2
    elif "return" in text:
        index = 1
    elif "ship" in text:
        index = 0
    elif "support" in text:
        index = 4
    else:
        return [0.2] * 5

    return [
        1.0 if i == index else 0.0
        for i in range(5)
    ]


class PipelineTests(unittest.TestCase):

    def setUp(self):
        # Create an isolated knowledge base.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.docs = Path(self.temp.name)

        for name, content in POLICIES.items():
            (self.docs / name).write_text(
                content,
                encoding="utf-8",
            )

        # Create an isolated ChromaDB collection.
        client = chromadb.EphemeralClient()

        self.collection = client.create_collection(
            name=f"pipeline_{uuid4().hex[:16]}",
            embedding_function=None,
            configuration={
                "hnsw": {"space": "cosine"}
            },
        )

        # Mock the embedding provider.
        self.embedder = Mock()

        self.embedder.embed_documents.side_effect = (
            lambda texts: [
                fake_vector(text)
                for text in texts
            ]
        )

        self.embedder.embed_query.side_effect = fake_vector

        # Mock Gemini response generation.
        self.client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(
                    side_effect=self.fake_generation
                )
            )
        )

        self.bot = RAGChatbot(
            self.collection,
            self.embedder,
            self.client,
        )

    def records(self):
        """Prepare records from the temporary knowledge base."""

        return prepare_chunks(self.docs)

    def index(self):
        """Index the temporary knowledge base."""

        return sync_collection(
            self.collection,
            self.embedder,
            self.records(),
        )

    def fake_generation(self, **kwargs):
        """Simulate structured Gemini responses."""

        config = kwargs["config"]
        contents = kwargs["contents"]

        # Simulate conversational query rewriting.
        if config.system_instruction == REWRITE_PROMPT:
            request = json.loads(contents)

            query = request["current_question"]

            if query.lower() == "how can i track it?":
                query = "How can I track my order?"

            return SimpleNamespace(
                text=json.dumps({"query": query})
            )

        # Parse the generation prompt.
        data = json.loads(contents.split("\n", 1)[1])

        question = data["standalone_question"].lower()

        # Select a simulated answer.
        if "bitcoin" in question:
            source, answer = None, None

        elif "track" in question:
            source = "tracking.md"
            answer = "Track your order from your store account."

        elif "refund" in question:
            source = "refunds.md"
            answer = "Approved refunds take 5-7 business days."

        elif "return" in question:
            source = "returns.md"
            answer = "Eligible returns are accepted within 30 days."

        elif "ship" in question:
            source = "shipping.md"
            answer = "Standard shipping takes 3-5 business days."

        else:
            source, answer = None, None

        # Cite only sources actually present in retrieved excerpts.
        matching = [
            f"S{i}"
            for i, part in enumerate(data["excerpts"], 1)
            if part["source"] == source
        ]

        supported = bool(matching)

        return SimpleNamespace(
            text=json.dumps({
                "supported": supported,
                "answer": answer if supported else "",
                "source_ids": matching[:1],
            })
        )

    def test_indexes_all_five_documents_with_sources(self):
        result = self.index()

        self.assertEqual(result["stored"], 5)

        stored = self.collection.get(
            include=["metadatas"]
        )

        sources = {
            item["source"]
            for item in stored["metadatas"]
        }

        self.assertEqual(sources, set(POLICIES))

    def test_shipping_query_ranks_shipping_first(self):
        self.index()

        found = retrieve(
            "How long is shipping?",
            self.collection,
            self.embedder,
        )

        self.assertEqual(
            found[0]["source"],
            "shipping.md",
        )

    def test_refund_query_ranks_refunds_first(self):
        self.index()

        found = retrieve(
            "When is my refund?",
            self.collection,
            self.embedder,
        )

        self.assertEqual(
            found[0]["source"],
            "refunds.md",
        )

    def test_generated_answer_cites_retrieved_source(self):
        self.index()

        answer = self.bot.chatbot(
            "How long does shipping take?"
        )

        self.assertIn("3-5 business days", answer)

        self.assertIn(
            "Sources: shipping.md (chunk 0)",
            answer,
        )

        self.assertEqual(len(self.bot.history), 1)

    def test_followup_rewrites_and_uses_tracking_source(self):
        self.index()

        self.bot.chatbot(
            "How long does shipping take?"
        )

        response = self.bot.chatbot(
            "How can I track it?"
        )

        self.assertIn(
            "Sources: tracking.md",
            response,
        )

        self.assertEqual(
            self.embedder.embed_query.call_args.args[0],
            "How can I track my order?",
        )

        self.assertEqual(len(self.bot.history), 2)

    def test_reset_blocks_ambiguous_retrieval(self):
        self.index()

        self.bot.chatbot(
            "How long does shipping take?"
        )

        self.bot.reset_history()

        self.embedder.embed_query.reset_mock()
        self.client.models.generate_content.reset_mock()

        answer = self.bot.chatbot(
            "How long does it take?"
        )

        self.assertEqual(
            answer,
            CLARIFICATION_ANSWER,
        )

        self.embedder.embed_query.assert_not_called()

        self.client.models.generate_content.assert_not_called()

        self.assertEqual(self.bot.history, [])

    def test_unchanged_documents_do_not_reembed(self):
        self.index()

        self.embedder.embed_documents.reset_mock()

        result = self.index()

        self.assertEqual(result["written"], 0)
        self.assertEqual(result["unchanged"], 5)

        self.embedder.embed_documents.assert_not_called()

    def test_updated_document_replaces_old_chunk(self):
        self.index()

        old = {
            row["id"]
            for row in self.records()
            if row["metadata"]["source"] == "shipping.md"
        }

        (self.docs / "shipping.md").write_text(
            "# Shipping Policy\n"
            "Standard shipping takes 4-6 business days after dispatch.",
            encoding="utf-8",
        )

        result = self.index()

        ids = set(
            self.collection.get(include=[])["ids"]
        )

        self.assertEqual(result["deleted"], 1)

        self.assertTrue(old.isdisjoint(ids))

        self.assertEqual(self.collection.count(), 5)

    def test_deleted_document_is_removed_from_collection(self):
        self.index()

        (self.docs / "returns.md").unlink()

        result = self.index()

        sources = {
            item["source"]
            for item in self.collection.get(
                include=["metadatas"]
            )["metadatas"]
        }

        self.assertEqual(result["deleted"], 1)

        self.assertNotIn("returns.md", sources)

        self.assertEqual(self.collection.count(), 4)

    def test_unsupported_question_returns_fallback(self):
        self.index()

        answer = self.bot.chatbot(
            "Can I pay with Bitcoin?"
        )

        self.assertEqual(
            answer,
            UNSUPPORTED_ANSWER,
        )

        self.assertEqual(
            self.bot.history[-1]["answer"],
            UNSUPPORTED_ANSWER,
        )

    def test_empty_collection_avoids_all_api_requests(self):
        answer = self.bot.chatbot(
            "How long does shipping take?"
        )

        self.assertEqual(
            answer,
            UNSUPPORTED_ANSWER,
        )

        self.embedder.embed_query.assert_not_called()

        self.client.models.generate_content.assert_not_called()

    def test_generation_failure_does_not_change_history(self):
        self.index()

        self.client.models.generate_content.side_effect = (
            RuntimeError("API unavailable")
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "API unavailable",
        ):
            self.bot.chatbot(
                "How long does shipping take?"
            )

        self.assertEqual(self.bot.history, [])


if __name__ == "__main__":
    unittest.main()