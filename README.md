# CodeAlpha FAQ Chatbot

## Task
CodeAlpha Artificial Intelligence - Task 2: Chatbot for FAQs.

## Features
- FAQ dataset containing questions and answers
- Text cleaning/preprocessing
- TF-IDF vectorization
- Cosine similarity matching
- Best matching FAQ response
- Similarity threshold to avoid unrelated answers
- Simple chat interface
- Suggested questions

## Technology
- Python
- Flask
- Scikit-learn
- HTML, CSS, JavaScript
- NLP preprocessing

## How to run
1. Install Python 3.10+.
2. Open this folder in VS Code.
3. Open the terminal.
4. Run:
   `pip install -r requirements.txt`
5. Run:
   `python app.py`
6. Open `http://127.0.0.1:5000` in your browser.

## How it works
1. The FAQ questions are stored in `app.py`.
2. User text is converted to lowercase and punctuation is removed.
3. TF-IDF converts the FAQ questions into numerical vectors.
4. The user question is converted into a vector.
5. Cosine similarity compares the user question with all FAQ questions.
6. The FAQ with the highest similarity is selected.
7. If the score is too low, the chatbot asks the user to rephrase the question.
