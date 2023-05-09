import sys

# We have to include this to whoever is the "__main__"
from model_training import OpinionSpamDetectorModel

# import flask app but need to call it "application" for WSGI to work
from back_end import app as application  # noqa
