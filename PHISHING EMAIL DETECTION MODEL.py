
# ============================================================
#          PHISHING EMAIL DETECTION MODEL
#          Python + Scikit-learn
# ============================================================

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. DATASET (INCLUDED IN THIS FILE)
# Labels: phishing or safe
# ============================================================

emails = [
    # Phishing examples
    ("URGENT! Verify your account password immediately.", "phishing"),
    ("You have won a prize. Click here to claim now.", "phishing"),
    ("Your bank account is suspended. Confirm your login.", "phishing"),
    ("Reset your password due to unusual account activity.", "phishing"),
    ("Verify your account now to avoid suspension.", "phishing"),
    ("Claim your free reward before this offer expires.", "phishing"),
    ("Click here to confirm your banking information.", "phishing"),
    ("Your account requires urgent verification.", "phishing"),
    ("Your password expires today. Login to restore access.", "phishing"),
    ("Congratulations! You are a winner. Claim your prize now.", "phishing"),
    ("Unusual activity detected. Verify your identity immediately.", "phishing"),
    ("Your account is locked. Confirm your details.", "phishing"),

    # Safe examples
    ("Please attend the school meeting tomorrow at 10 AM.", "safe"),
    ("Here are the notes from today's class.", "safe"),
    ("Your appointment is scheduled for Monday morning.", "safe"),
    ("Thank you for sending the assignment.", "safe"),
    ("Can we meet at the library this afternoon?", "safe"),
    ("Your order receipt is attached to this email.", "safe"),
    ("The team meeting has been moved to Friday.", "safe"),
    ("Please review the report before tomorrow's meeting.", "safe"),
    ("Happy birthday! Hope you have a wonderful day.", "safe"),
    ("The homework deadline is next Wednesday.", "safe"),
    ("Thank you for attending today's presentation.", "safe"),
    ("I have attached the notes from our discussion.", "safe"),
]

# ============================================================
# 2. LOAD DATASET
# ============================================================

data = pd.DataFrame(emails, columns=["text", "label"])

X = data["text"]
y = data["label"]

print("=" * 55)
print("             PHISHING EMAIL DETECTOR")
print("=" * 55)

print("\nDataset loaded successfully!")
print("Total emails:", len(data))
print("\nDataset distribution:")
print(data["label"].value_counts())

# ============================================================
# 3. EXTRACT ADDITIONAL EMAIL FEATURES
# ============================================================

def extract_features(email_list):
    features = []

    suspicious_keywords = [
        "urgent", "verify", "password", "account",
        "winner", "prize", "click here", "suspended",
        "login", "bank", "confirm", "free",
        "limited time", "reset password",
        "unusual activity"
    ]

    for email in email_list:
        text = str(email).lower()

        # Count URLs
        urls = re.findall(
            r"(?:https?://|www\.)[^\s<>\"']+",
            text
        )

        # Count suspicious keywords
        keyword_count = sum(
            text.count(word) for word in suspicious_keywords
        )

        # Detect URLs containing an IPv4 address
        ip_url_count = len(re.findall(
            r"https?://(?:\d{1,3}\.){3}\d{1,3}(?=[:/ \s]|$)",
            text
        ))

        features.append([
            len(text),
            len(urls),
            keyword_count,
            text.count("!"),
            text.count("@"),
            ip_url_count,
            int("http://" in text),
            int("https://" in text),
            int("click here" in text),
            int("verify your account" in text)
        ])

    return csr_matrix(features, dtype=float)

# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining emails:", len(X_train))
print("Testing emails:", len(X_test))

# ============================================================
# 5. CONVERT TEXT INTO NUMERICAL FEATURES USING TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)

# Learn vocabulary only from training emails
X_train_text = vectorizer.fit_transform(X_train)
X_test_text = vectorizer.transform(X_test)

# Extract additional features
X_train_extra = extract_features(X_train)
X_test_extra = extract_features(X_test)

# Combine text and numerical features
X_train_final = hstack(
    [X_train_text, X_train_extra],
    format="csr"
)

X_test_final = hstack(
    [X_test_text, X_test_extra],
    format="csr"
)

# ============================================================
# 6. TRAIN THE MACHINE LEARNING MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_final, y_train)

print("\nModel training completed successfully!")

# ============================================================
# 7. EVALUATE THE MODEL
# ============================================================

y_pred = model.predict(X_test_final)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("                 MODEL EVALUATION")
print("=" * 55)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    labels=["phishing", "safe"],
    zero_division=0
))

# ============================================================
# 8. DISPLAY THE CONFUSION MATRIX
# ============================================================

labels = ["phishing", "safe"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Phishing", "Safe"],
    yticklabels=["Phishing", "Safe"]
)

plt.title("Phishing Email Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.show()

# ============================================================
# 9. PREDICT WHETHER A NEW EMAIL IS PHISHING OR SAFE
# ============================================================

def predict_email(email):

    # Convert email into numerical features
    text_features = vectorizer.transform([email])
    extra_features = extract_features([email])

    combined_features = hstack(
        [text_features, extra_features],
        format="csr"
    )

    # Predict the email class
    prediction = model.predict(combined_features)[0]

    # Calculate model probability estimates
    probabilities = model.predict_proba(combined_features)[0]
    classes = list(model.classes_)

    phishing_score = probabilities[
        classes.index("phishing")
    ]

    print("\n" + "-" * 55)
    print("                    EMAIL ANALYSIS")
    print("-" * 55)

    print("\nEmail:", email)
    print("Prediction:", prediction.upper())

    print(
        f"Estimated phishing score: {phishing_score * 100:.2f}%"
    )

    if prediction == "phishing":
        print("\nWARNING: Potential phishing email detected!")
    else:
        print("\nNo phishing detected by the model.")

    print(
        "\nNote: This is a machine learning prediction, "
        "not a guarantee of email safety."
    )

# ============================================================
# 10. INTERACTIVE EMAIL TESTING
# ============================================================

print("\n" + "=" * 55)
print("              TEST YOUR OWN EMAIL")
print("=" * 55)

while True:

    email_input = input(
        "\nPaste an email to analyze (or type 'exit' to quit):\n> "
    ).strip()

    if email_input.lower() == "exit":
        print("\nPhishing Email Detector closed.")
        break

    if not email_input:
        print("Please enter some email text.")
        continue

    predict_email(email_input)
