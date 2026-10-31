import unittest
from bot import chatbot


class TestKeywordChatbot(unittest.TestCase):

    def test_refund(self):
        result = chatbot("Can I get a refund?")
        expected = "Refunds are processed within 5–7 business days."
        self.assertEqual(result, expected)

    def test_shipping(self):
        result = chatbot("What is your shipping policy?")
        expected = "Standard shipping takes 3–5 business days."
        self.assertEqual(result, expected)

    def test_delivery(self):
        result = chatbot("When will my delivery arrive?")
        expected = "Standard shipping takes 3–5 business days."
        self.assertEqual(result, expected)

    def test_order(self):
        result = chatbot("Where is my order?")
        expected = "You can track your order through your account."
        self.assertEqual(result, expected)

    def test_package(self):
        result = chatbot("Where is my package?")
        expected = "You can track your order through your account."
        self.assertEqual(result, expected)

    def test_return(self):
        result = chatbot("I want to return an item")
        expected = "You can return products within 30 days."
        self.assertEqual(result, expected)

    def test_support(self):
        result = chatbot("I need customer support")
        expected = "Contact us at support@example.com."
        self.assertEqual(result, expected)

    def test_help(self):
        result = chatbot("I need help")
        expected = "Contact us at support@example.com."
        self.assertEqual(result, expected)

    def test_uppercase(self):
        result = chatbot("REFUND")
        expected = "Refunds are processed within 5–7 business days."
        self.assertEqual(result, expected)

    def test_whitespace(self):
        result = chatbot("  refund  ")
        expected = "Refunds are processed within 5–7 business days."
        self.assertEqual(result, expected)

    def test_multiple_keywords(self):
        result = chatbot("I have questions about shipping and refunds")

        expected = (
            "Refunds are processed within 5–7 business days.\n"
            "Standard shipping takes 3–5 business days."
        )

        self.assertEqual(result, expected)

    def test_multiple_categories(self):
        result = chatbot("I need help with my order")

        expected = (
            "You can track your order through your account.\n"
            "Contact us at support@example.com."
        )

        self.assertEqual(result, expected)

    def test_unknown_input(self):
        result = chatbot("What is your favorite color?")
        expected = "Sorry, I don't understand your request."
        self.assertEqual(result, expected)

    def test_empty_input(self):
        result = chatbot("")
        expected = "Sorry, I don't understand your request."
        self.assertEqual(result, expected)

    def test_no_duplicate_response(self):
        result = chatbot("shipping and delivery")
        expected = "Standard shipping takes 3–5 business days."
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()