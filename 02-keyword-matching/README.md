# Level 2: Keyword-Based Chatbot

## Overview

This project implements a **keyword-based customer support chatbot** using Python.

Unlike a rule-based chatbot that requires exact input matching, this chatbot searches for predefined keywords within user messages.

This allows it to recognize customer requests even when they are phrased differently.

The chatbot supports multiple keyword detection and can return several relevant responses when a message contains keywords from different categories.

This is the second level of the [Chatbot Levels](../README.md) project.

## How It Works

The chatbot follows this process:

```text
User Input
    |
    v
Input Normalization
    |
    v
Keyword Detection
    |
    v
Check All Keyword Categories
    |
    +---- Matches Found ----> Collect Responses
    |                              |
    |                              v
    |                       Combine Responses
    |
    +---- No Matches ------> Fallback Response
```

### 1. Input Normalization

The chatbot converts user input to lowercase and removes leading and trailing whitespace.

```python
message = message.lower().strip()
```

Example:

```text
Input:  "  WHERE IS MY ORDER?  "
Output: "where is my order?"
```

This ensures that keyword detection is case-insensitive.

### 2. Keyword Detection

The chatbot uses Python's `in` operator to check whether specific keywords appear within a message.

```python
if "refund" in message:
    responses.append("Refunds are processed within 5–7 business days.")
```

Unlike exact string matching, the `in` operator searches for a substring anywhere within the message.

For example:

```text
Input: Can I get a refund?
Detected Keyword: refund
```

The chatbot recognizes the keyword even though the entire message does not match a predefined command.

### 3. Multiple Keyword Detection

The chatbot uses independent `if` statements instead of an `if/elif` chain.

```python
if "refund" in message:
    responses.append("Refund information")

if "shipping" in message:
    responses.append("Shipping information")
```

This allows multiple conditions to match within the same message.

For example:

```text
User: I have questions about shipping and refunds

Bot: Refunds are processed within 5–7 business days.
Standard shipping takes 3–5 business days.
```

### 4. Response Collection

The chatbot stores matching responses in a Python list.

```python
responses = []
```

When a keyword matches, its predefined response is added to the list.

```python
responses.append("Refunds are processed within 5–7 business days.")
```

After checking all conditions, the chatbot combines the responses:

```python
return "\n".join(responses)
```

Each response appears on a separate line.

### 5. Fallback Handling

If no keywords match, the chatbot returns a fallback response.

```python
if not responses:
    return "Sorry, I don't understand your request."
```

This prevents the chatbot from returning an empty response.

## Supported Keywords

The chatbot recognizes the following keywords:

| Category | Keywords | Description |
|---|---|---|
| Refunds | `refund` | Provides refund information |
| Shipping | `shipping`, `delivery` | Provides shipping information |
| Orders | `order`, `package` | Provides order-tracking information |
| Returns | `return` | Explains the return policy |
| Support | `support`, `help` | Provides customer support contact information |

The chatbot supports five response categories using nine predefined keywords.

All store policies and contact information are fictional and provided for demonstration purposes.

## Getting Started

### Requirements

- Python 3.8 or later
- No external dependencies

The chatbot uses only built-in Python features.

### Run the Chatbot

From the repository's root directory:

```bash
python 02-keyword-matching/bot.py
```

Alternatively:

```bash
cd 02-keyword-matching
python bot.py
```

### Example Conversation

```text
Keyword-Based Customer Support Chatbot
Type 'exit' to quit.

You: Can I get a refund?
Bot: Refunds are processed within 5–7 business days.

You: Where is my package?
Bot: You can track your order through your account.

You: When will my delivery arrive?
Bot: Standard shipping takes 3–5 business days.

You: I need help with shipping and returns
Bot: Standard shipping takes 3–5 business days.
You can return products within 30 days.
Contact us at support@example.com.

You: hello
Bot: Sorry, I don't understand your request.

You: exit
Bot: Goodbye!
```

## Project Structure

```text
02-keyword-matching/
├── bot.py          # Keyword-based chatbot implementation
├── test_bot.py     # Automated unit tests
└── README.md       # Project documentation
```

### File Descriptions

**bot.py**

Contains the main chatbot implementation.

- `chatbot(message)`: Normalizes input, detects keywords, and returns matching responses.
- `main()`: Runs the interactive command-line interface.

**test_bot.py**

Contains automated tests using Python's built-in `unittest` framework.

## Testing

The project includes 15 automated unit tests.

These tests verify the chatbot's behavior across different customer support scenarios.

### Run Tests

From the `02-keyword-matching` directory:

```bash
python -m unittest discover -v
```

### Test Coverage

| Test | Description |
|---|---|
| `test_refund` | Recognizes refund requests |
| `test_shipping` | Recognizes shipping requests |
| `test_delivery` | Recognizes delivery-related keywords |
| `test_order` | Recognizes order-related questions |
| `test_package` | Recognizes package-related questions |
| `test_return` | Recognizes return requests |
| `test_support` | Recognizes customer support requests |
| `test_help` | Recognizes help requests |
| `test_uppercase` | Handles uppercase input |
| `test_whitespace` | Handles leading and trailing whitespace |
| `test_multiple_keywords` | Recognizes multiple categories in one message |
| `test_multiple_categories` | Combines order and support responses |
| `test_unknown_input` | Returns a fallback for unknown input |
| `test_empty_input` | Handles empty messages |
| `test_no_duplicate_response` | Avoids duplicate responses for synonyms in one category |

The expected result is 15 passing tests with no failures.

## Comparison: Level 1 vs. Level 2

| Feature | Level 1: Rule-Based | Level 2: Keyword-Based |
|---|---|---|
| Exact input matching | Yes | Not required |
| Keyword detection | No | Yes |
| Alternative phrasing | Limited | Supported when keywords match |
| Multiple categories per message | No | Yes |
| Input normalization | Yes | Yes |
| Machine learning | No | No |
| Natural language understanding | No | No |
| Conversation memory | No | No |

### Example

**Level 1**

```text
User: Can I get a refund?

Bot: Sorry, I don't understand your request.
```

**Level 2**

```text
User: Can I get a refund?

Bot: Refunds are processed within 5–7 business days.
```

Level 2 improves flexibility by detecting keywords rather than requiring exact phrases.

## Limitations

Although keyword matching is more flexible than exact string matching, this approach still has several limitations.

### 1. No Context Understanding

The chatbot recognizes keywords but does not understand the meaning of a sentence.

For example:

```text
User: I don't want a refund.

Bot: Refunds are processed within 5–7 business days.
```

The chatbot responds because it detects the word `refund`, despite the customer's negative statement.

### 2. Substring Matching

The chatbot uses substring matching rather than complete-word matching.

For example:

```python
"refund" in "refunded"
```

Returns:

```text
True
```

This can be useful, but it can also cause false positives when keywords appear inside unrelated words.

### 3. Limited Vocabulary

The chatbot only recognizes predefined keywords.

For example:

```text
User: How long does postage take?
```

Because `postage` is not included in the keyword list, the chatbot returns its fallback response.

### 4. No Intent Classification

The chatbot cannot reliably determine what the user wants.

For example:

```text
User: I need to cancel my order.
```

The chatbot detects `order` and returns tracking information, even though the user wants to cancel an order.

### 5. No Conversation Memory

Each message is processed independently.

The chatbot cannot remember previous messages or understand follow-up questions.

### 6. Fixed Response Ordering

When several keyword categories match, responses appear in the order defined by the source code.

The chatbot does not determine which response is most relevant.

## Key Concepts Learned

This implementation demonstrates:

- Keyword-based chatbot architecture
- Substring matching with Python's `in` operator
- Input normalization
- Independent conditional statements
- Lists and response collection
- Handling multiple keyword matches
- Fallback response handling
- Automated unit testing with `unittest`
- Limitations of keyword-based natural language processing

## Next Level: Intent-Based Chatbot

The next implementation introduces **intent classification**.

Instead of simply detecting predefined keywords, the chatbot will attempt to identify the purpose behind a user's message.

For example:

```text
User: I want to cancel my purchase.

Predicted Intent: cancel_order
```

An intent-based chatbot can use machine learning techniques such as:

- Text preprocessing
- TF-IDF vectorization
- Logistic Regression classification
- Confidence-based fallback handling

This approach will allow the chatbot to recognize patterns across different ways of expressing similar requests.

---

**Part of the [Chatbot Levels](../README.md) learning project.**