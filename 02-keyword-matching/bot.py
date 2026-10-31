def chatbot(message):
    message = message.lower().strip()
    responses = []

    if "refund" in message:
        responses.append("Refunds are processed within 5–7 business days.")

    if "shipping" in message or "delivery" in message:
        responses.append("Standard shipping takes 3–5 business days.")

    if "order" in message or "package" in message:
        responses.append("You can track your order through your account.")

    if "return" in message:
        responses.append("You can return products within 30 days.")

    if "support" in message or "help" in message:
        responses.append("Contact us at support@example.com.")

    if not responses:
        return "Sorry, I don't understand your request."

    return "\n".join(responses)


def main():
    print("Keyword-Based Customer Support Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower().strip() == "exit":
            print("Bot: Goodbye!")
            break

        response = chatbot(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()