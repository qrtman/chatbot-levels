"""Offline tests for grounded response generation."""

import json
import sys
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import RAGChatbot, UNSUPPORTED_ANSWER


class BotTests(unittest.TestCase):

    def setUp(self):
        self.collection = Mock()
        self.embedder = Mock()
        self.client = Mock()

        self.chunks = [
            {
                "id": "shipping-0",
                "source": "shipping.md",
                "chunk_index": 0,
                "text": "Standard shipping takes 3–5 business days.",
            },
            {
                "id": "returns-1",
                "source": "returns.md",
                "chunk_index": 1,
                "text": "Returns are accepted within 30 calendar days.",
            },
        ]

        patcher = patch(
            "bot.retrieve",
            return_value=self.chunks,
        )

        self.retrieve_mock = patcher.start()
        self.addCleanup(patcher.stop)

        self.bot = RAGChatbot(
            self.collection,
            self.embedder,
            self.client,
        )

    def mock_reply(
        self,
        supported=True,
        answer="Shipping takes 3–5 business days.",
        ids=None,
    ):
        if ids is None:
            ids = ["S1"]

        payload = {
            "supported": supported,
            "answer": answer,
            "source_ids": ids,
        }

        self.client.models.generate_content.return_value = (
            SimpleNamespace(text=json.dumps(payload))
        )

    def test_grounded_answer_has_only_cited_source(self):
        self.mock_reply()

        answer = self.bot.chatbot("  How long is shipping?  ")

        self.assertIn(
            "Shipping takes 3–5 business days.",
            answer,
        )

        self.assertIn(
            "Sources: shipping.md (chunk 0)",
            answer,
        )

        self.assertNotIn("returns.md", answer)

        self.retrieve_mock.assert_called_once_with(
            "How long is shipping?",
            self.collection,
            self.embedder,
            top_k=4,
        )

    def test_prompt_includes_question_and_labeled_evidence(self):
        self.mock_reply()

        self.bot.chatbot("How long is shipping?")

        prompt = (
            self.client.models.generate_content
            .call_args.kwargs["contents"]
        )

        self.assertIn("How long is shipping?", prompt)
        self.assertIn("Standard shipping takes 3", prompt)
        self.assertIn('"id": "S1"', prompt)
        self.assertIn('"id": "S2"', prompt)

    def test_uses_structured_generation_and_model(self):
        self.mock_reply()

        self.bot.chatbot("Shipping?")

        kwargs = (
            self.client.models.generate_content
            .call_args.kwargs
        )

        self.assertEqual(
            kwargs["model"],
            "gemini-3.5-flash-lite",
        )

        self.assertEqual(
            kwargs["config"].response_mime_type,
            "application/json",
        )

        self.assertTrue(
            kwargs["config"].system_instruction
        )

    def test_multiple_sources(self):
        self.mock_reply(ids=["S2", "S1"])

        answer = self.bot.chatbot("What are the policies?")

        self.assertIn(
            "Sources: returns.md (chunk 1), "
            "shipping.md (chunk 0)",
            answer,
        )

    def test_duplicate_sources_are_removed(self):
        self.mock_reply(ids=["S1", "S1"])

        answer = self.bot.chatbot("Shipping?")

        self.assertEqual(
            answer.count("shipping.md"),
            1,
        )

    def test_empty_retrieval_does_not_call_gemini(self):
        self.retrieve_mock.return_value = []

        answer = self.bot.chatbot("Shipping?")

        self.assertEqual(answer, UNSUPPORTED_ANSWER)

        self.client.models.generate_content.assert_not_called()

    def test_empty_question_does_not_call_services(self):
        answer = self.bot.chatbot("   ")

        self.assertEqual(
            answer,
            "Please enter a question.",
        )

        self.retrieve_mock.assert_not_called()

    def test_non_string_question_is_rejected(self):
        answer = self.bot.chatbot(None)

        self.assertEqual(
            answer,
            "Please enter a question.",
        )

        self.retrieve_mock.assert_not_called()

    def test_unsupported_response_uses_safe_fallback(self):
        self.mock_reply(
            supported=False,
            answer="Invented guarantee",
            ids=[],
        )

        answer = self.bot.chatbot("Do you sell cars?")

        self.assertEqual(answer, UNSUPPORTED_ANSWER)

    def test_invalid_json_raises(self):
        self.client.models.generate_content.return_value = (
            SimpleNamespace(text="not json")
        )

        with self.assertRaises(RuntimeError):
            self.bot.chatbot("Shipping?")

    def test_unknown_source_id_raises(self):
        self.mock_reply(ids=["S99"])

        with self.assertRaisesRegex(
            RuntimeError,
            "unknown source",
        ):
            self.bot.chatbot("Shipping?")

    def test_supported_response_needs_citations(self):
        self.mock_reply(ids=[])

        with self.assertRaises(RuntimeError):
            self.bot.chatbot("Shipping?")

    def test_wrong_field_types_raise(self):
        self.client.models.generate_content.return_value = (
            SimpleNamespace(
                text=json.dumps({
                    "supported": "true",
                    "answer": "X",
                    "source_ids": "S1",
                })
            )
        )

        with self.assertRaises(RuntimeError):
            self.bot.chatbot("Shipping?")

    def test_gemini_failure_propagates(self):
        self.client.models.generate_content.side_effect = (
            RuntimeError("API unavailable")
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "API unavailable",
        ):
            self.bot.chatbot("Shipping?")


if __name__ == "__main__":
    unittest.main()