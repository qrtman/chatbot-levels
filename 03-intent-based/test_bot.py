import unittest

from bot import (
    IntentChatbot,
    FALLBACK_RESPONSE,
    load_intents,
    prepare_training_data,
)


class TestIntentChatbot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Train the model once for the entire test suite.
        cls.bot = IntentChatbot()

    def test_load_intents(self):
        intents = load_intents()
        self.assertEqual(len(intents), 6)

    def test_training_data(self):
        intents = load_intents()

        texts, labels, responses = prepare_training_data(intents)

        self.assertEqual(len(texts), 60)
        self.assertEqual(len(labels), 60)
        self.assertEqual(len(responses), 6)

    def test_model_trained(self):
        self.assertIsNotNone(self.bot.model)

    def test_greeting_intent(self):
        intent, confidence = self.bot.predict_intent("hello")

        self.assertEqual(intent, "greeting")

    def test_order_intent(self):
        intent, confidence = self.bot.predict_intent(
            "where is my order"
        )

        self.assertEqual(intent, "track_order")

    def test_support_intent(self):
        intent, confidence = self.bot.predict_intent(
            "i need to speak to a human"
        )

        self.assertEqual(intent, "contact_support")

    def test_refund_raw_prediction(self):
        # Disable rejection to inspect the underlying prediction.
        bot = IntentChatbot(confidence_threshold=0.0)

        intent, confidence = bot.predict_intent(
            "can I get a refund?"
        )

        self.assertEqual(intent, "refund")

    def test_confidence_range(self):
        intent, confidence = self.bot.predict_intent("hello")

        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_empty_input(self):
        response = self.bot.chatbot("")

        self.assertEqual(response, FALLBACK_RESPONSE)

    def test_whitespace_input(self):
        response = self.bot.chatbot("   ")

        self.assertEqual(response, FALLBACK_RESPONSE)

    def test_greeting_response(self):
        response = self.bot.chatbot("hello")

        self.assertEqual(
            response,
            "Hello! How can I help you today?"
        )

    def test_order_response(self):
        response = self.bot.chatbot("where is my order")

        self.assertEqual(
            response,
            "You can track your order through your account."
        )

    def test_custom_threshold(self):
        bot = IntentChatbot(confidence_threshold=1.0)

        response = bot.chatbot("hello")

        self.assertEqual(response, FALLBACK_RESPONSE)

    def test_response_exists_for_every_intent(self):
        for intent in self.bot.model.classes_:
            with self.subTest(intent=intent):
                self.assertIn(intent, self.bot.responses)


if __name__ == "__main__":
    unittest.main()