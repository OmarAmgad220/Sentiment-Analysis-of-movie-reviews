# imports
import numpy as np
from flask import Flask, request, render_template
import pickle
from project import preprocess_text
from flask import redirect, url_for

app = Flask(__name__)

# load the feature extractor
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

# load the models

models = {
        "SVM": pickle.load(open('models/SVM.pkl', 'rb')),
        "LogisticRegression": pickle.load(open('models/Logistic_Regression.pkl', 'rb')),
        "NaiveBayes": pickle.load(open('models/Naive_Bayes.pkl', 'rb')),
        "RandomForest": pickle.load(open('models/Random_Forest.pkl', 'rb')),
        "KNN": pickle.load(open('models/KNN.pkl', 'rb')),
}  

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
    model_name = request.args.get('model')
    if not model_name:
        # Redirect to default model if not provided in URL
        return redirect(url_for('home', model='SVM'))

    return render_template('index.html', usedModel=model_name, positive=positive, negative=negative)



# model selection
@app.route('/select', methods=['POST'])
def select():
    model_name = request.form.get('models', 'SVM')
    return redirect(url_for('home', model=model_name))  # redirect to home with model in query



# predict results
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model_name = request.form.get('usedModel', 'SVM')
        input_text = request.form['content']
        preprocessed_input_text = preprocess_text(input_text)
        features = vectorizer.transform([preprocessed_input_text])

        # Use the selected model from the dictionary
        selected_model = models.get(model_name, models['SVM'])  # Fallback to SVM if model not found
        prediction = selected_model.predict(features)[0]

        if prediction == 1:
            positive.append(input_text + ' - ' + model_name)
            pickle.dump(positive, open('data/positive.pkl', 'wb'))
            return render_template('index.html', positive=positive, negative=negative, predict='Positive', usedModel=model_name)
        else:
            negative.append(input_text + ' - ' + model_name)
            pickle.dump(negative, open('data/negative.pkl', 'wb'))
            return render_template('index.html', positive=positive, negative=negative, predict='Negative', usedModel=model_name)


if __name__ == "__main__":
    app.run()