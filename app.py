from flask import Flask, render_template, request, jsonify
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

FAQS = [
    {"question": "What is CodeAlpha?", "answer": "CodeAlpha is a software development and internship platform that provides practical learning opportunities."},
    {"question": "What is an AI internship?", "answer": "An AI internship gives practical experience with artificial intelligence, machine learning, NLP, model development, and related workflows."},
    {"question": "How many tasks are required?", "answer": "The internship instructions require a minimum of two tasks, and the task list says interns can complete any two or three of the four AI tasks."},
    {"question": "Where should I upload my source code?", "answer": "The instructions say to upload the complete source code to GitHub in a repository named CodeAlpha_ProjectName."},
    {"question": "Do I need to post my project on LinkedIn?", "answer": "Yes. The instructions ask interns to post a video explanation of the project on LinkedIn with the GitHub repository link."},
    {"question": "What is the submission method?", "answer": "The completed task must be submitted through the submission form shared in the respective WhatsApp group."},
    {"question": "What certificate can I receive?", "answer": "The internship overview lists an Internship Completion Certificate, a QR Verified Completion Certificate, and a Unique ID Certificate. A Letter of Recommendation may also be provided based on performance."},
    {"question": "What is a chatbot?", "answer": "A chatbot is a software application that interacts with users through messages and provides automated responses."},
    {"question": "What is NLP?", "answer": "Natural Language Processing (NLP) is a field of AI that helps computers process and understand human language."},
    {"question": "What is cosine similarity?", "answer": "Cosine similarity measures how similar two text vectors are by comparing the angle between them. A higher value means the texts are more similar."},
    {"question": "What is TF-IDF?", "answer": "TF-IDF is a text representation technique that gives higher importance to words that are useful for distinguishing documents."},
    {"question": "What is object detection?", "answer": "Object detection identifies objects in an image or video and usually returns their locations using bounding boxes and labels."}
]

def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9\s]", " ", text)

questions = [clean_text(item["question"]) for item in FAQS]
vectorizer = TfidfVectorizer(stop_words="english")
faq_matrix = vectorizer.fit_transform(questions)

def find_best_answer(user_question):
    cleaned = clean_text(user_question)
    if not cleaned.strip():
        return "Please type a question.", 0.0

    user_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(user_vector, faq_matrix)[0]
    best_index = scores.argmax()
    best_score = float(scores[best_index])

    # Prevent random answers when the similarity is too low.
    if best_score < 0.18:
        return "Sorry, I could not find a close FAQ match. Please ask a question related to the internship, AI, NLP, or this project.", best_score

    return FAQS[best_index]["answer"], best_score

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    answer, score = find_best_answer(question)
    return jsonify({"answer": answer, "similarity": round(score, 3)})

if __name__ == "__main__":
    app.run(debug=True)
