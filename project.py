import os
import pickle
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from wordcloud import STOPWORDS
import string
from nltk.stem import WordNetLemmatizer
import contractions
from nltk.corpus import wordnet
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def load_reviews(positive_folder, negative_folder):
    reviews = []
    labels = []
    
    # Read positive reviews
    for filename in os.listdir(positive_folder):
            file_path = os.path.join(positive_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                review_text = file.read().strip()
                reviews.append(review_text)
                labels.append('positive')

    # Read negative reviews
    for filename in os.listdir(negative_folder):
            file_path = os.path.join(negative_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                review_text = file.read().strip()
                reviews.append(review_text)
                labels.append('negative')
    
    df = pd.DataFrame({'review_text': reviews, 'sentiment': labels})
    
    return df


lemmatizer = WordNetLemmatizer()

def lemmatization_text(sentence):
    
	def pos_tagger(nltk_tag):
		if nltk_tag.startswith('J'):
			return wordnet.ADJ
		elif nltk_tag.startswith('V'):
			return wordnet.VERB
		elif nltk_tag.startswith('N'):
			return wordnet.NOUN
		elif nltk_tag.startswith('R'):
			return wordnet.ADV
		else:
			return None

	pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
 
	wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

	lemmatized_sentence = []
	for word, tag in wordnet_tagged:
		if tag is None:
			lemmatized_sentence.append(word)
		else:
			lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
	lemmatized_sentence = " ".join(lemmatized_sentence)

	return lemmatized_sentence


def preprocess_text(text):
    
    stop_words = set(STOPWORDS.intersection(stopwords.words('english')))-{"not","very","wow", "oh",  
    "yuck","yay", "alas", "oops", "hey", "ah", "ouch", "phew","damn","hell","ugh",
    "jeez", "gosh", "wowza", "whoa", "yikes", "meh", "huh", "yay!", "ugh!", "oh no","nothing"}
    text = contractions.fix(text)
    
    text = text.lower()
    
    text = text.translate(str.maketrans('', '', ''.join(set(string.punctuation) - {'!'})))
    
    # with lemmatization
    # tokens = lemmatization_text(text)
    # tokens = nltk.word_tokenize(tokens)
    
    # without lemmatization
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)


def main():
    
    # region for loading data and preprocessing
	
    # Paths (to read the raw data txt files)
	
    # pos_folder = 'pos'
    # neg_folder = 'neg'
    # reviews_df = load_reviews(pos_folder, neg_folder)
	
    reviews_df = pd.read_csv("movie_reviews.csv")
    reviews_df['review_text'] = reviews_df['review_text'].apply(preprocess_text)
    reviews_df['sentiment'] = reviews_df['sentiment'].map({'positive': 1, 'negative': 0})
    
    #TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=2000)
    X = tfidf_vectorizer.fit_transform(reviews_df['review_text'])
    y = reviews_df['sentiment']
	
    # endregion
    
    # region for trainning and testing
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Logistic Regression classifier
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    print("Logistic Performance:")
    print(f"Accuracy: { 100 * accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Save To new CSV file
    # reviews_df.to_csv('processed_movie_reviews.csv', index=False)
    # print("\nProcessed DataFrame saved to 'processed_movie_reviews.csv'")
    
    # SVM classifier
    model = SVC(kernel='linear',C=1.6)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    print("SVM Performance:")
    print(f"Accuracy: {100 * accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    # Save To new CSV file
    # reviews_df.to_csv('processed_movie_reviews.csv', index=False)
    # print("\nProcessed DataFrame saved to 'processed_movie_reviews.csv'")
    
    # endregion
	
    # region Complete model/s and example prediction
	
    CompleteLogistic = LogisticRegression(max_iter=1000)
    CompleteLogistic.fit(X, y)
    CompleteSVM = SVC(kernel='linear',C=1.6)
    CompleteSVM.fit(X, y)
    CompleteNB = MultinomialNB()
    CompleteNB.fit(X, y)
    CompleteRF = RandomForestClassifier(n_estimators=100, random_state=42)
    CompleteRF.fit(X, y)
    CompleteKNN = KNeighborsClassifier(n_neighbors=5)
    CompleteKNN.fit(X, y)
    
    input_text = "it was a great movie"
    input_text = preprocess_text(input_text)
    features = tfidf_vectorizer.transform([input_text])
    prediction = CompleteLogistic.predict(features)[0]
    print(f"Prediction of Logistic for '{input_text}': {'Positive' if prediction == 1 else 'Negative'}")
    prediction = CompleteSVM.predict(features)[0]
    print(f"Prediction of SVM for '{input_text}': {'Positive' if prediction == 1 else 'Negative'}")
    
    # pickle.dump(CompleteLogistic, open('models/Logistic_Regression.pkl', 'wb'))
    # pickle.dump(CompleteSVM, open('models/SVM.pkl', 'wb')) 
    # pickle.dump(CompleteNB, open('models/Naive_Bayes.pkl', 'wb'))
    # pickle.dump(CompleteRF, open('models/Random_Forest.pkl', 'wb'))
    # pickle.dump(CompleteKNN, open('models/KNN.pkl', 'wb'))
    # pickle.dump(tfidf_vectorizer, open('models/tfidf_vectorizer.pkl', 'wb'))
	
    #endregion

if __name__ == "__main__":
    main()