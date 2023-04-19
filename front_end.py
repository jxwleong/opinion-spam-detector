from flask import Flask, render_template, request
import pickle
import json
# Need to import the pickled object
# else when load will have this problem
# AttributeError: Can't get attribute 'OpinionSpamDetectorModel' on <module '__main__' from '.\\main.py'>
from model_training import OpinionSpamDetectorModel
# load the model from disk
filename = 'finalized_model.bin'
with open(filename, 'rb') as f:
    loaded_model = pickle.load(f)



mapping = {
    "OR": "Authentic",
    "CG": "Spam"
}
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        # Do something with the text here
        response = 'You entered: ' + text
        return render_template('index.html', response=response)
    else:
        return render_template('index.html')
    
@app.route('/process_text', methods=['POST'])
def process_text():
    text = request.form['text']
    # Do something with the text here

    new_review = text
    new_review_vect = loaded_model.vectorizer.transform([new_review])
    new_review_pred = loaded_model.model.predict(new_review_vect)
    print("Prediction for new review:", new_review_pred[0])
    response_data = {
        "review": text,
        "outcome": new_review_pred[0],
        "result": mapping[new_review_pred[0]]
    }
    response = json.dumps(response_data, indent=4)
    return render_template('index.html', response=response)


if __name__ == '__main__':
    app.run(debug=True)