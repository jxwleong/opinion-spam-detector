__author__ = ["jxwleong"]

import os
import sys 

REPO_ROOT= os.path.normpath(os.path.join(os.path.abspath(__file__), ".."))
LIB_PATH = os.path.join(REPO_ROOT, "lib")
DATASET_PATH = os.path.join(REPO_ROOT, "dataset", "fake reviews dataset.csv")
sys.path.insert(0, LIB_PATH)

# Import modules from lib folder
import pandas as pd
from lib.sklearn.model_selection import train_test_split
from lib.sklearn.feature_extraction.text import TfidfVectorizer
from lib.sklearn.linear_model import LogisticRegression
from lib.sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load CSV file into pandas dataframe
df = pd.read_csv(DATASET_PATH)
print(df)
# Separate input data from labels
X = df["review_text"]
y = df["is_fake_review"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess input data using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Evaluate model performance on testing data
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Use model to predict new reviews
new_review = "This product is amazing!"
new_review_vect = vectorizer.transform([new_review])
new_review_pred = model.predict(new_review_vect)
print("Prediction for new review:", new_review_pred[0])