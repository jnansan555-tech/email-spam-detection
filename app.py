import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load dataset
data = pd.read_csv("combined_data.csv")

# Select required columns
data = data[["label", "text"]]

# Remove missing values
data = data.dropna()

print("Total emails:", len(data))
print(data.head())


# 2. Separate input and output
X = data["text"]  
y = data["label"]     


# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# 4. Convert text into numbers using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)


# 5. Create and train Naive Bayes model
model = MultinomialNB()

model.fit(X_train, y_train)


# 6. Test the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 7. Predict a new email
def check_email(email):

    email = tfidf.transform([email])

    prediction = model.predict(email)

    if prediction[0] == 1:
        return "SPAM"
    else:
        return "HAM"


# Test emails
email2 = "Congratulations! You won a free iPhone. Click here to claim now!"
email1 = "Hi team, please attend the meeting tomorrow."

print("\nEmail 1:", check_email(email1))
print("Email 2:", check_email(email2))

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

# Create bar chart
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [accuracy, precision, recall, f1]

print("\nAccuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1-score :", round(f1 * 100, 2), "%")


plt.bar(metrics, [v * 100 for v in values])

plt.title("Email Spam Detection - Model Performance")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score (%)")
plt.ylim(0, 100)

plt.savefig("results.png", dpi=150, bbox_inches="tight", facecolor="white")   # save first
plt.show()                                                                     # then show