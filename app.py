# imports
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# load the feature extractor and models
LogisticRegression = pickle.load(open('models/Logistic_Regression.pkl', 'rb'))

# load the saved vectorizer
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# load positive and negative arrays
try:
    positive = pickle.load(open('data/positive.pkl', 'rb'))
    negative = pickle.load(open('data/negative.pkl', 'rb'))
except:
    positive = []
    negative = []

# home menu
@app.route('/')
def home():
    return render_template('index.html', positive=positive, negative=negative, predict = '')


# predict results
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['content']
        features = vectorizer.transform([input_text])
        prediction = LogisticRegression.predict(features)[0]

        if prediction == 1:
            positive.append(input_text)
            pickle.dump(positive, open('data/positive.pkl', 'wb'))
            return render_template('index.html', positive=positive, negative=negative, predict='Positive')
        else:
            negative.append(input_text)
            pickle.dump(negative, open('data/negative.pkl', 'wb'))
            return render_template('index.html', positive=positive, negative=negative, predict='Negative')

if __name__ == "__main__":
    app.run()