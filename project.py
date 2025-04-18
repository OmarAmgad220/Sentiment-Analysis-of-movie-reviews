import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


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
    
    # Create DataFrame
    df = pd.DataFrame({'review_text': reviews, 'sentiment': labels})
    
    return df

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove punctuation, tokenize, remove stopwords.
    """
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main():
    # Replace these paths with your actual folder paths
    pos_folder = 'pos'
    neg_folder = 'neg'
    
    # Load reviews
    reviews_df = load_reviews(pos_folder, neg_folder)
    
    # Preprocess reviews
    reviews_df['processed_text'] = reviews_df['review_text'].apply(preprocess_text)
    
    # Convert sentiments to binary (positive=1, negative=0)
    reviews_df['sentiment_label'] = reviews_df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Apply TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=2000)  # Limit to top 5000 features
    X = tfidf_vectorizer.fit_transform(reviews_df['processed_text'])
    y = reviews_df['sentiment_label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Logistic Regression classifier
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = classifier.predict(X_test)
    
    # Evaluate the model
    print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Save the DataFrame with processed data
    reviews_df.to_csv('processed_movie_reviews.csv', index=False)
    print("\nProcessed DataFrame saved to 'processed_movie_reviews.csv'")

if __name__ == "__main__":
    main()