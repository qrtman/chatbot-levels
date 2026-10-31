# Level 1: Rule-Based Chatbot

## Overview

This project implements a simple **rule-based customer support chatbot** using Python.

The chatbot responds to user messages by comparing them against predefined rules. Each recognized input produces a fixed response.

This is the first level of the **Chatbot Levels** project, which explores how chatbot systems evolve from basic conditional logic to more advanced AI-powered approaches.

## How It Works

The chatbot follows a straightforward process:

```text
User Input
    |
    v
Input Normalization
    |
    v
Exact String Matching
    |
    +---- Match Found ----> Return Predefined Response
    |
    +---- No Match -------> Return Fallback Response
```

### 1. Input Normalization

Before processing a message, the chatbot converts the input to lowercase and removes leading and trailing whitespace.

```python
message = message.lower().strip()
```

For example:

```text
Input:  "  HELLO  "
Output: "hello"
```

This allows the chatbot to recognize inputs regardless of capitalization or surrounding spaces.

### 2. Rule Matching

The chatbot uses Python's `if`, `elif`, and `else` statements to match user messages against predefined rules.

```python
if message == "hello":
    return "Hello! Welcome to customer support."

elif message == "track order":
    return "You can track your order through your account."
```

Each rule checks whether the normalized user input exactly matches a predefined string.

### 3. Fallback Response

If the chatbot cannot find a matching rule, it returns a default response.

```python
else:
    return "Sorry, I don't understand your request."
```

This ensures that the chatbot always provides a response, even when it cannot understand the user's message.

## Supported Commands

The chatbot currently supports the following customer support requests:

| User Input | Function |
|---|---|
| `hello` | Greets the customer |
| `track order` | Provides order-tracking information |
| `refund policy` | Explains refund processing times |
| `shipping information` | Provides shipping information |
| `return item` | Explains the return policy |
| `contact support` | Provides customer support contact information |
| `exit` | Ends the conversation |

All store policies and contact information are fictional and used for demonstration purposes.

## Getting Started

### Requirements

- Python 3.8 or later
- No external libraries required

### Run the Chatbot

From the repository's root directory:

```bash
python 01-rule-based/bot.py
```

Alternatively, navigate into the directory:

```bash
cd 01-rule-based
python bot.py
```

### Example Conversation

```text
Customer Support Chatbot
Type 'exit' to quit.

You: hello
Bot: Hello! Welcome to customer support.

You: track order
Bot: You can track your order through your account.

You: refund policy
Bot: Refunds are processed within 5–7 business days.

You: Can I get a refund?
Bot: Sorry, I don't understand your request.

You: exit
Bot: Goodbye!
```

## Project Structure

```text
01-rule-based/
├── bot.py          # Chatbot implementation
├── test_bot.py     # Automated unit tests
└── README.md       # Project documentation
```

### File Descriptions

**bot.py**

Contains two main functions:

- `chatbot(message)`: Processes user input and returns a predefined response.
- `main()`: Runs the interactive command-line interface.

**test_bot.py**

Contains automated tests using Python's built-in `unittest` framework.

## Testing

The chatbot includes automated unit tests to verify its expected behavior.

### Run Tests

From the `01-rule-based` directory:

```bash
python -m unittest discover -v
```

### Implemented Tests

| Test | Description |
|---|---|
| `test_greeting` | Verifies the chatbot recognizes `hello` |
| `test_uppercase` | Verifies case-insensitive matching |
| `test_unknown_input` | Verifies fallback behavior for unknown messages |

### Test Results

All three tests passed during local testing.

```text
Ran 3 tests in 0.000s

OK
```

## Limitations

Although the chatbot handles predefined inputs successfully, it has several limitations.

### 1. Exact String Matching

The chatbot only recognizes messages that match its predefined rules.

For example:

```text
Input: refund policy
Result: Recognized
```

```text
Input: Can I get a refund?
Result: Not recognized
```

Both messages relate to refunds, but only the predefined phrase is recognized.

### 2. No Natural Language Understanding

The chatbot does not understand the meaning or intention behind user messages.

It only compares strings using conditional statements.

### 3. Limited Scalability

Adding new commands requires manually modifying the source code.

As the number of supported commands increases, maintaining numerous conditional statements becomes more difficult.

### 4. No Conversation Context

The chatbot treats every message independently.

It cannot remember previous messages or maintain a conversation history.

## Key Concepts Learned

This implementation demonstrates:

- Rule-based chatbot architecture
- Conditional statements (`if`, `elif`, `else`)
- String comparison and normalization
- Functions and separation of responsibilities
- Command-line user interaction
- Fallback response handling
- Automated unit testing with `unittest`

## Next Level: Keyword-Based Chatbot

The next implementation introduces **keyword matching** to overcome some limitations of exact string comparison.

Instead of requiring an exact phrase, the chatbot will search for relevant keywords within user messages.

For example:

```text
User: Can I get a refund?

Detected Keyword: refund

Bot: Refunds are processed within 5–7 business days.
```

This approach provides greater flexibility without requiring machine learning.

However, keyword matching still has limitations when interpreting ambiguous messages or understanding context.

These limitations will motivate the development of the intent-based chatbot in Level 3.

---

**Part of the [Chatbot Levels](../README.md) learning project.**