from flask import Flask, render_template, request, jsonify
import pickle
import json
import os 



mapping = {
    "OR": "Authentic",
    "CG": "Spam"
}


model_path = {
    "SVM": os.path.join(os.path.dirname(__file__), "model", 'final_trained_model_SVM.bin'),
    "Logistic Regression": os.path.join(os.path.dirname(__file__), "model", 'final_trained_model_Logistic_Regression.bin'),
    "KNN": os.path.join(os.path.dirname(__file__), "model", 'final_trained_model_KNN.bin'),
    "Decision Tree": os.path.join(os.path.dirname(__file__), "model", 'final_trained_model_Decision_Tree.bin'),
    "Random Forest": os.path.join(os.path.dirname(__file__), "model", 'final_trained_model_Random_Forest.bin'),
}

# Setup default model
current_model_path = model_path["SVM"]
for model_name, path in model_path.items():
    if path == current_model_path:
        current_model_name = model_name
        break

with open(current_model_path, "rb") as f:
    model, vectorizer  = pickle.load(f)

app = Flask(__name__)

@app.template_filter('json_format')
def json_format(value):
    return json.dumps(json.loads(value), indent=4)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_name=current_model_name,  model_path=model_path)

    
@app.route("/api/change_model", methods=["POST"])
def change_model():
    global current_model
    global current_model_name
    global model 
    global vectorizer
    
    model_name = request.form["model_name"]
    current_model_name = model_name
    if model_name in model_path:
        current_model_path = model_path[model_name]
        with open(current_model_path, "rb") as f:
            model, vectorizer  = pickle.load(f)
        return render_template("index.html", model_name=model_name, model_path=model_path)
    else:
        return jsonify({"message": f"Model {model_name} not found"}), 400
    


@app.route("/api/predict", methods=["GET", "POST"])
def predict():
    # Parse the input data from the request payload
    #data = request.get_json(force=True)
    #text = data['text']
    if request.method == "POST":
        text = request.form["text"]
    else:
        text = request.args.get("text", "")

    new_review = text
    new_review_vect = vectorizer.transform([new_review])
    new_review_pred = model.predict(new_review_vect)

    response_data = {
        "model": current_model_name,
        "text": text,
        "outcome": new_review_pred[0],
        "result": mapping[new_review_pred[0]]
    }

    response = json.dumps(response_data, indent=4)

    if request.method == "POST":
        return render_template("index.html", response=response, model_name=current_model_name, model=model, model_path=model_path)
    else:
        return response

# Route for getting the logs
@app.route("/api/logs", methods=["GET"])
def get_logs():
    with open("main.log", "r") as f:
        logs = f.read().splitlines()
    return jsonify(logs)

@app.route("/api/logs/print", methods=["POST"])
def print_logs():
    with open("main.log", "r") as f:
        logs = f.read()
    return render_template("logs.html", logs=logs)

# Route for getting the models metric
@app.route("/api/metrics", methods=["GET", "POST"])
def get_model_metrics():
    metrics_json = os.path.join(os.path.dirname(__file__), "model", 'model_metrics.json')
    
    with open(metrics_json, "r") as f:
        metrics = json.load(f)
        formatted_metrics = json.dumps(metrics, indent=4)
        
    if request.method == "GET":
        return formatted_metrics
    else:
        return render_template("index.html", response=formatted_metrics, model_name=current_model_name, model=model, model_path=model_path)

if __name__ == "__main__":
    app.run(debug=True)