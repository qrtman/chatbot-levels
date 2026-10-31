def chatbot(message):
    message = message.lower().strip()

    if message == "hello":
        return "Hello! Welcome to customer support."

    elif message == "track order":
        return "You can track your order through your account."

    elif message == "refund policy":
        return "Refunds are processed within 5–7 business days."

    elif message == "shipping information":
        return "Standard shipping takes 3–5 business days."

    elif message == "return item":
        return "You can return products within 30 days."

    elif message == "contact support":
        return "Contact us at support@example.com."

    else:
        return "Sorry, I don't understand your request."


def main():
    print("Customer Support Chatbot")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")

        if user_input.lower().strip() == "exit":
            print("Bot: Goodbye!")
            break

        response = chatbot(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()