import unittest
from bot import chatbot


class TestChatbot(unittest.TestCase):

    def test_greeting(self):
        result = chatbot("HELLO")

        expected = "Hello! Welcome to customer support."

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()