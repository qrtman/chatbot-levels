import unittest
from bot import chatbot


class TestChatbot(unittest.TestCase):

    def test_greeting(self):
        result = chatbot("hello")
        expected = "Hello! Welcome to customer support."
        self.assertEqual(result, expected)

    def test_uppercase(self):
        result = chatbot("HELLO")
        expected = "Hello! Welcome to customer support."
        self.assertEqual(result, expected)

    def test_unknown_input(self):
        result = chatbot("aaasdqfgriughaosd")
        expected = "Sorry, I don't understand your request."
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()