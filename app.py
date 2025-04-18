# imports
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# load the feature extractor and models
# featureExtractor = pickle.load(open('models/featureExtractionVectorizer.pkl', 'rb'))

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
    return render_template('index.html', positive=positive, negative=negative)


# predict results
@app.route('/predict',methods=['POST'])
def predict():
    return render_template('index.html', positive=positive, negative=negative)


if __name__ == "__main__":
    app.run()