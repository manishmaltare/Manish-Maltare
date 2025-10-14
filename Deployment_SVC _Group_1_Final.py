import os
import pickle
import re
import string
import numpy as np
import pandas as pd
import requests
import nltk
import streamlit as st
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.svm import SVC
import math

df = pd.read_csv('dataset -P582.csv')

# Ensure that nltk resources are available
nltk.download('punkt', quiet=True)

# --- STEP 1: Define Augmented Stop Words, Romanized Mapping, and Negation Handler ---
hindi_noise_words = [
    'ka', 'ki', 'ke', 'aur', 'hai', 'main', 'mai', 'bhi', 'yeh', 'ye',
    'ko', 'mein', 'me', 'to', 'ho', 'hona', 'tha', 'thi', 'the',
    'nhi', 'fir', 'phir', 'lekin', 'par', 'apne', 'usko', 'kuch', 'hoga',
    'kya', 'tera', 'mera', 'diya', 'diye', 'de'
]

romanized_mapping = {
    'accha': 'good', 'acha': 'good', 'achha': 'good', 'bahut': 'very',
    'bhut': 'very', 'kharab': 'bad', 'bekar': 'bad', 'sahi': 'correct',
    'nhi': 'not', 'nahi': 'not', 'milega': 'receive', 'mila': 'received',
    'liya': 'take', 'diya': 'give', 'lene': 'take', 'achcha':'good',
    'achhi':'good', 'achchaa':'good', 'acchi':'good', 'aap':'you',
    'bakwaas':'bad', 'bakwas':'bad', 'bada':'big'
}

# Negation handling words (extending the previous list)
negation_words = [
    "not", "no", "never", "none", "neither", "nobody", "nothing", "nowhere",
    "can't", "won't", "couldn't", "wouldn't", "isn't", "wasn't", "don't",
    "doesn't", "didn't", "aren't", "weren't", "hasn't", "haven't", "wouldn't",
    "may not", "might not", "no one", "no more", "no longer", "none at all",
    "nobody else", "nothing else", "no way", "not at all", "not really", "not quite"
]

# Define a function to handle negation (mark negated words with _NEG)
def handle_negation(text):
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
    new_tokens = []
    negate = False
    
    for token in tokens:
        if token in negation_words:
            negate = True
            new_tokens.append(token)
        elif negate and re.match(r"\w+", token):
            new_tokens.append(token + "_NEG")  # Mark the word as negated
            negate = False
        else:
            new_tokens.append(token)
    
    return " ".join(new_tokens)

# Load stopwords from GitHub (as already implemented)
STOPWORDS_URL = "https://github.com/manishmaltare/Manish-Maltare/blob/main/english_stopwords.txt"
def load_stopwords_from_github():
    response = requests.get(STOPWORDS_URL)
    if response.status_code == 200:
        stopwords_list = response.text.splitlines()
        return set(stopwords_list)
    else:
        raise Exception(f"Failed to load stopwords from GitHub: {response.status_code}")

# Load the stopwords
english_stopwords = load_stopwords_from_github()

# Combine English stopwords with Hindi noise words
stop_words = english_stopwords.union(set(hindi_noise_words))

# --- STEP 2: Preprocessing Function ---
def vectorized_clean_series(s: pd.Series):
    s = s.fillna('').astype(str)
    s = s.str.normalize('NFKC')
    s = s.str.lower()

    # 1. Clean URLs, emails, HTML tags, and numbers
    s = s.str.replace(r'http\S+|www\.\S+', ' ', regex=True)
    s = s.str.replace(r'\S+@\S+', ' ', regex=True)
    s = s.str.replace(r'<.*?>', ' ', regex=True)
    s = s.str.replace(r'\d+(?:[.,]\d+)*', ' ', regex=True)

    # 2. Replace Romanized words
    def replace_romanized_words(text):
        if not text:
            return ""
        words = text.split()
        words = [romanized_mapping.get(word, word) for word in words]
        return ' '.join(words)
    
    s = s.apply(replace_romanized_words)

    # 3. Remove punctuation and tokenization (using augmented stop words)
    s = s.str.replace(r'[^a-z\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    # 4. Tokenize and remove stopwords
    s = s.apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words and word]))

    return s

# --- STEP 3: Sentiment Lexicon and Score Functions ---
def load_lexicon(filename):
    lexicon = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                word, score = parts[0], parts[1]
                try:
                    lexicon[word] = float(score)
                except ValueError:
                    continue  # Skip malformed lines
    return lexicon

# Load sentiment lexicon
sentiment_lexicon = load_lexicon("lexicon.txt")

# Text preprocessing: lowercase, remove punctuation, strip spaces
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# Sentiment scoring
def get_sentiment_score(word):
    return sentiment_lexicon.get(word, 0)

# Compound sentiment score for input text
def calculate_compound_score(text):
    tokens = preprocess_text(text).split()
    return sum(get_sentiment_score(token) for token in tokens)

# Sentiment category from score
def get_sentiment_category(text):
    compound_score = calculate_compound_score(text)
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# --- STEP 4: Preprocessing Data and Model ---
# Assuming df_new_2['body'] is your input column, we apply the negation handler
df_new_2['body'] = df_new_2['body'].apply(handle_negation)

# --- STEP 5: Streamlit Interface ---
# Streamlit app for prediction and analysis
st.title("Sentiment Analysis with Negation Handling 😊😐😞")
st.subheader("SVC Model - Get sentiment probabilities with confidence levels")

# Load pre-trained model and vectorizer (TF-IDF)
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocess and predict sentiment
def preprocess_and_predict(text_list):
    cleaned_lines = vectorized_clean_series(pd.Series(text_list)).tolist()
    probas, classes = predict_sentiment_with_proba(cleaned_lines)
    return probas, classes

def predict_sentiment_with_proba(text_list):
    # Process the input text using pre-trained vocab and idf
    tfidf_feat = compute_tfidf(text_list, vocab, idf)
    probas = model.predict_proba(tfidf_feat)
    class_labels = model.classes_
    return probas, class_labels

# Streamlit input and output
text_input = st.text_area("Enter text (one or more lines)", height=150)

if st.button("Analyze Sentiment"):
    if text_input:
        raw_lines = [line.strip() for line in text_input.split('\n') if line.strip()]
        probabilities, classes = preprocess_and_predict(raw_lines)

        for i, line in enumerate(raw_lines):
            st.write(f"Input: **{line}**")
            
            # Find the index of highest probability for this input
            max_idx = probabilities[i].argmax()
            
            for idx, (label, prob) in enumerate(zip(classes, probabilities[i])):
                if idx == max_idx:
                    st.write(f"- **{label}: {prob*100:.2f}%**")
                else:
                    st.write(f"- {label}: {prob*100:.2f}%")
            st.write("---")
    else:
        st.write("Please enter text.")
