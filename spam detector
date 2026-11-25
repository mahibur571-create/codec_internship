"""
Spam Message Classifier
Classifies messages as SPAM or HAM (not spam) using TF-IDF features
and two classic ML models: Naive Bayes and a Support Vector Machine.

Usage:
    python spam_classifier.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

# 1. Load data: each row is a label (ham/spam) and the message text
df = pd.read_csv(DATA_URL, sep="\t", names=["label", "message"])
print(f"Dataset: {len(df)} messages\n{df['label'].value_counts()}\n")

# 2. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. Turn text into numbers with TF-IDF (word importance scores)
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)   # learn vocabulary from training data only
X_test_vec = vectorizer.transform(X_test)

# 4. Train both models and compare them
models = {"Naive Bayes": MultinomialNB(), "SVM": LinearSVC()}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    print(f"=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

# 5. Try the best model on new messages
best = models["SVM"]
samples = [
    "Congratulations! You've won a free iPhone. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been suspended. Call 09061234567 immediately.",
]
for msg, pred in zip(samples, best.predict(vectorizer.transform(samples))):
    print(f"[{pred.upper()}] {msg}")
