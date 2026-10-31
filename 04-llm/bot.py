import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Locate the repository's .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """
You are a helpful customer support chatbot for a fictional online store.

Store policies:
- Standard shipping takes 3-5 business days.
- Products can be returned within 30 days.
- Refunds are processed within 5-7 business days.
- Customers can track orders through their accounts.
- Human support is available at support@example.com.

Instructions:
- Respond politely and clearly.
- Keep answers concise.
- Remember previous messages in the conversation.
- Only use the store policies provided above.
- Do not invent order statuses, refund approvals, or store policies.
- If you cannot answer a question, suggest contacting human support.
"""


class LLMChatbot:

    def __init__(self, client=None):
        if client is None:
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is missing from your .env file."
                )

            client = genai.Client(api_key=api_key)

        self.client = client

        self.chat = self.client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=500,
            ),
        )

    def chatbot(self, message):
        message = message.strip()

        if not message:
            return "Please enter a message."

        response = self.chat.send_message(message)

        return response.text or "Sorry, I couldn't generate a response."


def main():
    bot = LLMChatbot()

    print("LLM Customer Support Chatbot")
    print(f"Model: {MODEL_NAME}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ")

            if user_input.lower().strip() == "exit":
                print("Bot: Goodbye!")
                break

            response = bot.chatbot(user_input)

            print("Bot:", response)

        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()