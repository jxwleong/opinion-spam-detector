__author__ = ["jxwleong"]

import os
import sys 
import pickle
<<<<<<< HEAD
import logging 
import json
=======
>>>>>>> 1f0e28d062008dcd2f93430d89ee09589882e665

REPO_ROOT= os.path.normpath(os.path.join(os.path.abspath(__file__), ".."))
LIB_PATH = os.path.join(REPO_ROOT, "lib")
DATASET_PATH = os.path.join(REPO_ROOT, "dataset", "fake reviews dataset.csv")
sys.path.insert(0, LIB_PATH)

# Import modules from lib folder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Create a logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler and set its level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a file handler and set its level to INFO
file_handler = logging.FileHandler('main.log')
file_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

model_metrics = {}
model_bin_name_prefix = "final_trained_model_"
=======
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

>>>>>>> 1f0e28d062008dcd2f93430d89ee09589882e665
class OpinionSpamDetectorModel:
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = dataset_path

    def train(self):
        # Load CSV file into pandas dataframe
        self.df = pd.read_csv(self.dataset_path)

        self.x = self.df["text_"]
        self.y = self.df["label"]
<<<<<<< HEAD
        logger.info(self.y)
=======
        print(self.y)
>>>>>>> 1f0e28d062008dcd2f93430d89ee09589882e665
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

<<<<<<< HEAD
        logger.info("Accuracy:", self.accuracy)
        logger.info("Precision:", self.precision)
        logger.info("Recall:", self.recall)
        logger.info("F1 score:", self.f1)


if __name__ == "__main__":
    """
=======
        print("Accuracy:", self.accuracy)
        print("Precision:", self.precision)
        print("Recall:", self.recall)
        print("F1 score:", self.f1)

if __name__ == "__main__":
>>>>>>> 1f0e28d062008dcd2f93430d89ee09589882e665
    my_model = OpinionSpamDetectorModel()
    my_model.train()
    # Use model to predict new reviews
    new_review = "This product really helps to make the cooking easier!"
    new_review_vect = my_model.vectorizer.transform([new_review])
    new_review_pred = my_model.model.predict(new_review_vect)
<<<<<<< HEAD
    logger.info("Prediction for new review:", new_review_pred[0])

    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    # Save the model to use later.
    filename = os.path.join(os.path.dirname(__file__), 'finalized_model.bin')
    with open(filename, 'wb') as f:
        pickle.dump(my_model, f)
 
    """

    df = pd.read_csv(DATASET_PATH)

    x = df["text_"]
    y = df["label"]
    #logger.info(y)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Preprocess input data using TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Train a logistic regression model
    logger.info(f"Initialization of Logical Regression")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vect, y_train)

    logger.info(f"Initialization of KNN")
    # Train a KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_vect, y_train)

    logger.info(f"Initialization of Decision Tree")
    # Train a decision tree model
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train_vect, y_train)

    logger.info(f"Initialization of Random Forest")
    # Train a Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_vect, y_train)

    # Train a SVM model
    logger.info(f"Initialization of SVM")
    svm_model = SVC()
    svm_model.fit(X_train_vect, y_train)

    # Evaluate model performance on testing data
    models = {
        'Logistic Regression': lr_model,
        'KNN': knn_model,
        'Decision Tree': tree_model,
        'Random Forest': rf_model,
        'SVM': svm_model
    }

    logger.info(models)
    for name, model in models.items():
        logger.info(f"\nTraining for {name}")
        y_pred = model.predict(X_test_vect)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        logger.info(f"{name} Model Metrics")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 score: {f1}")
        logger.info("-------------------------------------------------------")

        model_metrics[name] = {}
        model_metrics[name]["Accuracy"] = accuracy
        model_metrics[name]["Precision"] = precision
        model_metrics[name]["Recall"] = recall
        model_metrics[name]["F1 score"] = f1

        model_underscore_name = name.replace(" ", " ")
        model_bin = os.path.join(os.path.dirname(__file__), "model", f"{model_bin_name_prefix}{model_underscore_name}.bin")
        if os.path.exists(model_bin) is False:
            with open(model_bin, 'wb') as f:
                pickle.dump((model, vectorizer), f)
        
        model_metrics_file = os.path.join(os.path.dirname(__file__), "model", f"model_metrics.json")
        with open(model_metrics_file, "w") as f:
            json.dump(model_metrics, f, indent=4)

    """
    
    filename = os.path.join(os.path.dirname(__file__), 'finalized_model_no_class.bin')
    with open(filename, 'wb') as f:
        pickle.dump((lr_model, vectorizer), f)
    """
 
=======
    print("Prediction for new review:", new_review_pred[0])

    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    # Save the model to use later.
    filename = filename = os.path.join(os.path.dirname(__file__), 'finalized_model.bin')
    with open(filename, 'wb') as f:
        pickle.dump(my_model, f)
 
>>>>>>> 1f0e28d062008dcd2f93430d89ee09589882e665
