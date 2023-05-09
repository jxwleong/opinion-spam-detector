## Opinion Spam Detector

Opinion Spam Detector is a web application that can detect and categorize opinion scams, fake reviews, and misleading content across various online platforms. It aims to enhance the reliability and trustworthiness of online opinions and reviews by utilizing natural language processing techniques and user-generated feedback. By achieving this objective, the application empowers individuals and businesses to make better-informed decisions, safeguard their interests, and contribute to the overall integrity and credibility of opinion-sharing platforms.

## Installation

To install and run the Opinion Spam Detector web application, follow the steps below:

1. Clone the repository: `git clone https://github.com/jxwleong/opinion-spam-detector.git`
2. Navigate to the cloned directory: `cd opinion-spam-detector`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate` (for Linux/MacOS) or `venv\Scripts\activate` (for Windows)
5. Install the dependencies: `pip install -r requirements.txt`
6. Train the model: `python model_training.py`. The trained model will be saved in the "model" folder.
7. Run the application: `python back_end.py`
8. Access the application on `http://localhost:5000/` in your web browser.

## Usage

Opinion Spam Detector has the following functionalities:

1. Spam Detection: The application can detect and categorize opinion scams, fake reviews, and misleading content across various online platforms.
2. Get Classification Model Metrics: The application provides classification model metrics that show the accuracy, precision, recall, and F1 score of the classification model.
3. Change Classification Model: The application allows users to change the classification model used for spam detection.
4. Get Logs: The application provides access to the logs of the application.
5. Give Feedback: The application allows users to provide feedback on the classification results.

## Credits

Opinion Spam Detector is developed and maintained by [jxwleong](https://github.com/jxwleong).
