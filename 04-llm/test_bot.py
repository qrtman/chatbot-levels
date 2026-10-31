import unittest
from unittest.mock import MagicMock, patch

from bot import LLMChatbot, MODEL_NAME, SYSTEM_PROMPT


class TestLLMChatbot(unittest.TestCase):

    def setUp(self):
        # Create a fake Gemini API client
        self.mock_client = MagicMock()

        # Create a fake chat session
        self.mock_chat = MagicMock()
        self.mock_client.chats.create.return_value = self.mock_chat

        # Initialize the chatbot without using a real API key
        self.bot = LLMChatbot(client=self.mock_client)

    def test_chat_session_created(self):
        self.mock_client.chats.create.assert_called_once()

    def test_correct_model(self):
        arguments = self.mock_client.chats.create.call_args

        self.assertEqual(
            arguments.kwargs["model"],
            MODEL_NAME
        )

    def test_system_prompt(self):
        arguments = self.mock_client.chats.create.call_args
        config = arguments.kwargs["config"]

        self.assertEqual(
            config.system_instruction,
            SYSTEM_PROMPT
        )

    def test_valid_message(self):
        self.mock_chat.send_message.return_value.text = "Hello!"

        response = self.bot.chatbot("hi")

        self.assertEqual(response, "Hello!")

    def test_message_sent_to_gemini(self):
        self.mock_chat.send_message.return_value.text = "Hello!"

        self.bot.chatbot("hi")

        self.mock_chat.send_message.assert_called_once_with("hi")

    def test_empty_input(self):
        response = self.bot.chatbot("")

        self.assertEqual(response, "Please enter a message.")

        self.mock_chat.send_message.assert_not_called()

    def test_whitespace_input(self):
        response = self.bot.chatbot("   ")

        self.assertEqual(response, "Please enter a message.")

        self.mock_chat.send_message.assert_not_called()

    def test_input_normalization(self):
        self.mock_chat.send_message.return_value.text = "Hello!"

        self.bot.chatbot("  hello  ")

        self.mock_chat.send_message.assert_called_once_with("hello")

    def test_empty_model_response(self):
        self.mock_chat.send_message.return_value.text = None

        response = self.bot.chatbot("hello")

        self.assertEqual(
            response,
            "Sorry, I couldn't generate a response."
        )

    def test_multiple_messages(self):
        self.mock_chat.send_message.return_value.text = "Response"

        self.bot.chatbot("Where is my order?")
        self.bot.chatbot("How can I track it?")

        self.assertEqual(
            self.mock_chat.send_message.call_count,
            2
        )

        # Verify both messages used the same chat session
        self.mock_client.chats.create.assert_called_once()

    def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                LLMChatbot()

    def test_api_error_propagates(self):
        self.mock_chat.send_message.side_effect = RuntimeError(
            "API unavailable"
        )

        with self.assertRaises(RuntimeError):
            self.bot.chatbot("hello")


if __name__ == "__main__":
    unittest.main()