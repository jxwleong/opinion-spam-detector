__author__ = ["jxwleong"]

import os
import sys 
import pickle

# Need to import the pickled object
# else when load will have this problem
# AttributeError: Can't get attribute 'OpinionSpamDetectorModel' on <module '__main__' from '.\\main.py'>
from model_training import OpinionSpamDetectorModel
# load the model from disk
filename = 'finalized_model.bin'
with open(filename, 'rb') as f:
    loaded_model = pickle.load(f)


new_review = "This product really helps to make the cooking easier!"
new_review_vect = loaded_model.vectorizer.transform([new_review])
new_review_pred = loaded_model.model.predict(new_review_vect)
print("Prediction for new review:", new_review_pred[0])