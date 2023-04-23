__author__ = ["jxwleong"]

import os
import sys 
import pickle

REPO_ROOT= os.path.normpath(os.path.join(os.path.abspath(__file__), ".."))
LIB_PATH = os.path.join(REPO_ROOT, "lib")
DATASET_PATH = os.path.join(REPO_ROOT, "dataset", "fake reviews dataset.csv")
sys.path.insert(0, LIB_PATH)

# Import modules from lib folder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class OpinionSpamDetectorModel:
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = dataset_path

    def train(self):
        # Load CSV file into pandas dataframe
        self.df = pd.read_csv(self.dataset_path)

        self.x = self.df["text_"]
        self.y = self.df["label"]
        print(self.y)
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # Preprocess input data using TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.X_train_vect = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vect = self.vectorizer.transform(self.X_test)

        # Train a logistic regression model
        self.model = LogisticRegression()
        self.model.fit(self.X_train_vect, self.y_train)

        # Evaluate model performance on testing data
        self.y_pred = self.model.predict(self.X_test_vect)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred, average='micro')
        self.recall = recall_score(self.y_test, self.y_pred, average='micro')
        self.f1 = f1_score(self.y_test, self.y_pred, average='micro')

        print("Accuracy:", self.accuracy)
        print("Precision:", self.precision)
        print("Recall:", self.recall)
        print("F1 score:", self.f1)

if __name__ == "__main__":
    my_model = OpinionSpamDetectorModel()
    my_model.train()
    # Use model to predict new reviews
    new_review = "This product really helps to make the cooking easier!"
    new_review_vect = my_model.vectorizer.transform([new_review])
    new_review_pred = my_model.model.predict(new_review_vect)
    print("Prediction for new review:", new_review_pred[0])

    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    # Save the model to use later.
    filename = filename = os.path.join(os.path.dirname(__file__), 'finalized_model.bin')
    with open(filename, 'wb') as f:
        pickle.dump(my_model, f)
 
