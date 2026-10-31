# Level 3: Intent-Based Chatbot

## Overview

This project implements an intent-based customer support chatbot using Python and machine learning.

Unlike rule-based and keyword-based chatbots, this implementation uses a trained classifier to predict the user's intent from their message.

It uses TF-IDF for text vectorization and Logistic Regression for intent classification.

This is Level 3 of the [Chatbot Levels](../README.md) project.

## How It Works

The chatbot follows this pipeline:

```text
User Message
     |
     v
TF-IDF Vectorization
     |
     v
Logistic Regression
     |
     v
Intent Prediction
     |
     v
Confidence Threshold
     |
     +---- Accepted ----> Predefined Response
     |
     +---- Rejected ----> Fallback Response
```

### TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical features that can be processed by a machine-learning model.

The implementation uses word unigrams and bigrams:

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)
```

### Logistic Regression

Logistic Regression is used to classify messages into predefined intent categories.

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

The model learns associations between text features and intent labels using labeled training examples.

### Confidence Threshold

The chatbot uses a confidence threshold of `0.35`.

When the highest predicted probability is below this threshold, the chatbot returns a fallback response.

The threshold is an initial heuristic and has not been calibrated against a held-out evaluation dataset.

## Supported Intents

| Intent | Description |
|---|---|
| `greeting` | Greetings and introductions |
| `track_order` | Order tracking |
| `refund` | Refund requests |
| `shipping` | Shipping information |
| `return_item` | Product returns |
| `contact_support` | Human customer support |

The training dataset contains 60 examples across six intents.

All store policies and contact information are fictional.

## Project Structure

```text
03-intent-based/
├── bot.py
├── intents.json
├── test_bot.py
└── README.md
```

**bot.py:** Implements training, prediction, response selection, and the command-line interface.

**intents.json:** Stores training examples, intent labels, and predefined responses.

**test_bot.py:** Contains automated tests using Python's built-in `unittest` framework.

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
```

The main external dependency is scikit-learn.

## Running the Chatbot

From the repository root:

```bash
python 03-intent-based/bot.py
```

Example:

```text
Intent-Based Customer Support Chatbot
Type 'exit' to quit.

You: hello
Bot: Hello! How can I help you today?

You: where is my order
Bot: You can track your order through your account.

You: exit
Bot: Goodbye!
```

## Testing

Run the automated tests:

```bash
python -m unittest discover -s 03-intent-based -v
```

The test suite contains 14 tests covering:

- Loading the training dataset
- Preparing training examples
- Model initialization
- Intent prediction
- Confidence values
- Response generation
- Empty input handling
- Confidence threshold behavior
- Response mapping

These tests verify selected functionality and known examples. They do not measure generalization accuracy on unseen messages.

## Comparison of Chatbot Levels

| Feature | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| Exact string matching | Yes | No | No |
| Keyword detection | No | Yes | Learned text features |
| Machine learning | No | No | Yes |
| TF-IDF | No | No | Yes |
| Intent classification | No | No | Yes |
| Confidence threshold | No | No | Yes |
| Predefined responses | Yes | Yes | Yes |

## Limitations

### Small Training Dataset

The model uses only 60 training examples across six intent categories.

This limits its ability to generalize to unfamiliar sentences.

### Spelling Errors

Word-based TF-IDF is sensitive to misspellings.

For example, `"oredr"` may be less recognizable than `"order"`.

### Confidence Threshold

A correctly classified request may still be rejected if its predicted probability falls below the threshold.

### No Conversation Memory

The chatbot treats each message independently and cannot remember earlier messages.

### No Generative Responses

The classifier predicts an intent and retrieves a predefined response.

It does not generate new answers.

### No Production Evaluation

The project does not currently include a held-out evaluation dataset or calibrated confidence threshold.

## Key Concepts Learned

- Intent classification
- Labeled training datasets
- Text vectorization
- TF-IDF
- Logistic Regression
- Machine-learning pipelines
- Confidence-based fallback handling
- Python unit testing

## Conclusion

Level 3 demonstrates how machine learning can replace explicit keyword rules with learned relationships between text features and intent categories.

It also illustrates the challenges of small training datasets, confidence thresholds, and handling unfamiliar language.

---

**Part of the [Chatbot Levels](../README.md) learning project.**