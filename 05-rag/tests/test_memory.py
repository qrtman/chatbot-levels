"""Offline tests for multi-turn RAG conversation handling."""

import json
import sys
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from bot import (
    RAGChatbot,
    UNSUPPORTED_ANSWER,
    REWRITE_PROMPT,
    CLARIFICATION_ANSWER,
    is_ambiguous_question,
)


def reply(answer="Shipping takes 3-5 business days.", ids=None):
    if ids is None:
        ids = ["S1"]

    return SimpleNamespace(
        text=json.dumps({
            "supported": True,
            "answer": answer,
            "source_ids": ids,
        })
    )


def rewrite(query):
    return SimpleNamespace(
        text=json.dumps({"query": query})
    )


class MemoryTests(unittest.TestCase):

    def test_ambiguous_question_without_history(self):
        """Ambiguous questions must not trigger retrieval."""

        answer = self.bot.chatbot("How long does it take?")

        self.assertEqual(answer, CLARIFICATION_ANSWER)

        self.retrieve_mock.assert_not_called()
        self.client.models.generate_content.assert_not_called()

        self.assertEqual(self.bot.history, [])

    def test_ambiguity_after_memory_reset(self):
        """Reset must remove the context needed to resolve 'it'."""

        self.client.models.generate_content.return_value = reply()

        self.bot.chatbot("How long does shipping take?")

        self.bot.reset_history()

        self.retrieve_mock.reset_mock()
        self.client.models.generate_content.reset_mock()

        answer = self.bot.chatbot("How long does it take?")

        self.assertEqual(answer, CLARIFICATION_ANSWER)

        self.retrieve_mock.assert_not_called()
        self.client.models.generate_content.assert_not_called()

    def test_explicit_question_is_not_ambiguous(self):
        """Questions with a clear subject should proceed normally."""

        self.client.models.generate_content.return_value = reply()

        answer = self.bot.chatbot(
            "How long does standard shipping take?"
        )

        self.assertIn("shipping.md", answer)

        self.retrieve_mock.assert_called_once()


    def test_unresolved_rewritten_query_requests_clarification(self):
        """An unresolved follow-up must not reach retrieval."""

        self.client.models.generate_content.side_effect = [
            reply(),
            rewrite("How long does it take?"),
        ]

        self.bot.chatbot("What is your shipping policy?")

        self.retrieve_mock.reset_mock()

        answer = self.bot.chatbot("How long does it take?")

        self.assertEqual(answer, CLARIFICATION_ANSWER)
        self.retrieve_mock.assert_not_called()
        self.assertEqual(len(self.bot.history), 1)

    def setUp(self):
        self.collection = Mock()
        self.embedder = Mock()
        self.client = Mock()

        self.chunks = [{
            "id": "shipping-0",
            "source": "shipping.md",
            "chunk_index": 0,
            "text": "Shipping takes 3-5 business days.",
        }]

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

    def test_first_turn_skips_rewrite_and_adds_memory(self):
        self.client.models.generate_content.return_value = reply()

        self.bot.chatbot("How long is shipping?")

        self.assertEqual(
            self.client.models.generate_content.call_count,
            1,
        )

        self.assertEqual(
            self.retrieve_mock.call_args.args[0],
            "How long is shipping?",
        )

        self.assertEqual(
            self.bot.history[0]["question"],
            "How long is shipping?",
        )

    def test_followup_rewrites_before_retrieval(self):
        self.client.models.generate_content.side_effect = [
            reply(),
            rewrite("How can I track my shipped order?"),
            reply("Track your package in your account."),
        ]

        self.bot.chatbot("How long is shipping?")
        self.bot.chatbot("How can I track it?")

        self.assertEqual(
            self.retrieve_mock.call_args.args[0],
            "How can I track my shipped order?",
        )

        self.assertEqual(
            self.client.models.generate_content.call_count,
            3,
        )

        self.assertEqual(
            self.bot.history[1]["question"],
            "How can I track it?",
        )

    def test_followup_generation_receives_history_and_original_question(self):
        self.client.models.generate_content.side_effect = [
            reply(),
            rewrite("How can I track the shipment?"),
            reply(),
        ]

        self.bot.chatbot("How long is shipping?")
        self.bot.chatbot("How can I track it?")

        prompt = (
            self.client.models.generate_content
            .call_args.kwargs["contents"]
        )

        payload = json.loads(
            prompt.split("\n", 1)[1]
        )

        self.assertEqual(
            payload["question"],
            "How can I track it?",
        )

        self.assertEqual(
            payload["standalone_question"],
            "How can I track the shipment?",
        )

        self.assertEqual(
            payload["conversation_history_for_reference_only"][0]["question"],
            "How long is shipping?",
        )

    def test_rewrite_config_is_structured(self):
        self.client.models.generate_content.side_effect = [
            reply(),
            rewrite("Refund duration?"),
            reply(),
        ]

        self.bot.chatbot("Shipping?")
        self.bot.chatbot("How long does it take?")

        config = (
            self.client.models.generate_content
            .call_args_list[1].kwargs["config"]
        )

        self.assertEqual(
            config.response_mime_type,
            "application/json",
        )

        self.assertEqual(
            config.system_instruction,
            REWRITE_PROMPT,
        )

    def test_memory_is_bounded(self):
        def generate(**kwargs):
            if kwargs["config"].system_instruction == REWRITE_PROMPT:
                data = json.loads(kwargs["contents"])

                return rewrite(
                    data["current_question"]
                )

            return reply()

        self.client.models.generate_content.side_effect = generate

        for number in range(5):
            self.bot.chatbot(f"Question {number}")

        self.assertEqual(
            len(self.bot.history),
            3,
        )

        self.assertEqual(
            [turn["question"] for turn in self.bot.history],
            ["Question 2", "Question 3", "Question 4"],
        )

    def test_reset_clears_history_and_next_turn_skips_rewrite(self):
        self.client.models.generate_content.return_value = reply()

        self.bot.chatbot("Shipping?")

        self.bot.reset_history()

        self.assertEqual(self.bot.history, [])

        self.bot.chatbot("Returns?")

        self.assertEqual(
            self.client.models.generate_content.call_count,
            2,
        )

        self.assertEqual(
            self.retrieve_mock.call_args.args[0],
            "Returns?",
        )

    def test_unsupported_answer_is_stored(self):
        self.retrieve_mock.return_value = []

        answer = self.bot.chatbot("Unknown question")

        self.assertEqual(
            answer,
            UNSUPPORTED_ANSWER,
        )

        self.assertEqual(
            self.bot.history[0]["answer"],
            UNSUPPORTED_ANSWER,
        )

        self.client.models.generate_content.assert_not_called()

    def test_malformed_rewrite_does_not_change_memory(self):
        self.client.models.generate_content.side_effect = [
            reply(),
            SimpleNamespace(text="not-json"),
        ]

        self.bot.chatbot("Shipping?")

        before = list(self.bot.history)

        with self.assertRaisesRegex(
            RuntimeError,
            "query rewriting",
        ):
            self.bot.chatbot("How about it?")

        self.assertEqual(self.bot.history, before)
        self.assertEqual(self.retrieve_mock.call_count, 1)

    def test_blank_rewrite_does_not_change_memory(self):
        self.client.models.generate_content.side_effect = [
            reply(),
            rewrite(" "),
        ]

        self.bot.chatbot("Shipping?")

        with self.assertRaisesRegex(
            RuntimeError,
            "invalid rewritten query",
        ):
            self.bot.chatbot("How long?")

        self.assertEqual(len(self.bot.history), 1)

    def test_generation_error_does_not_save_failed_turn(self):
        self.client.models.generate_content.side_effect = [
            reply(),
            rewrite("How do I track a package?"),
            RuntimeError("API unavailable"),
        ]

        self.bot.chatbot("Shipping?")

        with self.assertRaisesRegex(
            RuntimeError,
            "API unavailable",
        ):
            self.bot.chatbot("How do I track it?")

        self.assertEqual(len(self.bot.history), 1)

    def test_rejects_invalid_history_limit(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    RAGChatbot(
                        self.collection,
                        self.embedder,
                        self.client,
                        memory_turns=limit,
                    )

    def test_empty_input_leaves_history_unchanged(self):
        answer = self.bot.chatbot("   ")

        self.assertEqual(
            answer,
            "Please enter a question.",
        )

        self.assertEqual(self.bot.history, [])

        self.client.models.generate_content.assert_not_called()
        self.retrieve_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()