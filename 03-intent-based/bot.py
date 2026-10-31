import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DATA_PATH = Path(__file__).with_name("intents.json")

FALLBACK_RESPONSE = "Sorry, I don't understand your request."


def load_intents():
    """Load the training examples from intents.json."""

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["intents"]


def prepare_training_data(intents):
    """Convert intent definitions into training examples and labels."""

    texts = []
    labels = []
    responses = {}

    for intent in intents:
        tag = intent["tag"]
        responses[tag] = intent["response"]

        for pattern in intent["patterns"]:
            texts.append(pattern)
            labels.append(tag)

    return texts, labels, responses


def train_model(texts, labels):
    """Train an intent classification model."""

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    model.fit(texts, labels)

    return model


class IntentChatbot:

    def __init__(self, confidence_threshold=0.35):
        intents = load_intents()

        texts, labels, responses = prepare_training_data(intents)

        self.model = train_model(texts, labels)
        self.responses = responses
        self.confidence_threshold = confidence_threshold

    def predict_intent(self, message):
        """Predict the most likely intent and its estimated probability."""

        probabilities = self.model.predict_proba([message])[0]

        best_index = probabilities.argmax()

        intent = self.model.classes_[best_index]
        confidence = float(probabilities[best_index])

        if confidence < self.confidence_threshold:
            return None, confidence

        return intent, confidence

    def chatbot(self, message):
        """Generate a response using the predicted intent."""

        if not message.strip():
            return FALLBACK_RESPONSE

        intent, confidence = self.predict_intent(message)

        if intent is None:
            return FALLBACK_RESPONSE

        return self.responses[intent]


def main():
    bot = IntentChatbot()

    print("Intent-Based Customer Support Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower().strip() == "exit":
            print("Bot: Goodbye!")
            break

        intent, confidence = bot.predict_intent(user_input)

        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.2f}")

        response = bot.chatbot(user_input)

        print("Bot:", response)



if __name__ == "__main__":
    main()