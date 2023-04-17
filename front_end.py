from flask import Flask, render_template, request

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
    response = 'You entered: ' + text
    return render_template('index.html', response=response)


if __name__ == '__main__':
    app.run(debug=True)