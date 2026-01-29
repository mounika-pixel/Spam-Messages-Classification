from flask import Flask, render_template, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Spam probability threshold
SPAM_THRESHOLD = 0.75

# Calls-to-action keywords that indicate spam
CTA_KEYWORDS = [
    "click", "buy", "order", "call now", "act now", "limited time", "offer expires",
    "claim your", "verify account", "confirm identity", "update payment", "download now",
    "install now", "subscribe now", "apply now", "register now", "sign up", "win",
    "congratulations won", "claim prize", "cash back", "free credit"
]

def is_short_neutral_message(message):
    """
    Check if message is short and neutral (likely NOT spam).
    Rules: word_count <= 4 AND no URLs AND no phone AND no money terms AND no CTAs
    """
    text = message.lower().strip()
    
    # Rule 1: Check word count (max 4 words)
    word_count = len(text.split())
    if word_count > 4:
        return False
    
    # Rule 2: Check for URLs
    if re.search(r'http|www|\.com|\.in|\.net|\.org', text):
        return False
    
    # Rule 3: Check for phone numbers
    if re.search(r'\b\d{7,}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\+\d+', text):
        return False
    
    # Rule 4: Check for currency/money terms
    if re.search(r'₹|£|€|dollar|rupee|price|cost|free|discount|offer|deal', text):
        return False
    
    # Rule 5: Check for calls-to-action
    for cta in CTA_KEYWORDS:
        if cta in text:
            return False
    
    return True


@app.route('/')
def home():
    return render_template("index.html", message="")


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = tfidf.transform([message])

    # Extract spam keywords
    keywords = [
        "free", "win", "winner", "claim", "prize", "urgent", "act now", "limited", "offer",
        "click", "link", "http", "www", "verify", "account", "password", "bank",
        "congratulations", "gift", "bonus", "guaranteed", "credit", "loan", "otp"
    ]
    lower_message = message.lower()
    found_keywords = [word for word in keywords if word in lower_message]
    found_keywords = list(dict.fromkeys(found_keywords))  # Remove duplicates, keep order

    probabilities = model.predict_proba(data)[0]
    spam_probability = probabilities[1]  # Probability of class 1 (spam)
    confidence = round(spam_probability * 100, 2)

    # Apply threshold-based classification (0.75)
    # Also check if short + neutral → force NOT SPAM
    if is_short_neutral_message(message):
        result = "Not Spam"
        confidence = round((1 - spam_probability) * 100, 2)
    elif spam_probability >= SPAM_THRESHOLD:
        result = "Spam Message"
    else:
        result = "Not Spam"

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        message=message,
        keywords=found_keywords
    )


# 🔥 Real-Time Email Text Analyzer API
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data['text']

    vector = tfidf.transform([text])
    prob = model.predict_proba(vector)
    spam_probability = prob[0][1]  # Probability of class 1 (spam)

    # Apply threshold and short message logic
    if is_short_neutral_message(text):
        final_spam_probability = 0  # Force to NOT spam
    elif spam_probability >= SPAM_THRESHOLD:
        final_spam_probability = spam_probability
    else:
        final_spam_probability = spam_probability

    return jsonify({
        "spam_probability": round(final_spam_probability * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
