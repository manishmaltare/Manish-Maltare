import os
os.system('pip install nltk')
import nltk
import requests
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import string
import streamlit as st
import re

df = pd.read_csv('dataset -P582.csv')

# --- STEP 1: DEFINE AUGMENTED STOP WORDS AND MAPPING DICTIONARY ---

# 1. Common Romanized Hindi Function Words (Noise)
hindi_noise_words = [
    'ka', 'ki', 'ke', 'aur', 'hai', 'main', 'mai', 'bhi', 'yeh', 'ye',
    'ko', 'mein', 'me', 'to', 'ho', 'hona', 'tha', 'thi', 'the',
    'nhi', 'fir', 'phir', 'lekin', 'par', 'apne', 'usko', 'kuch', 'hoga',
    'kya', 'tera', 'mera', 'diya', 'diye', 'de'
]

# 2. Key Romanized Sentiment/Functional Word Mapping
romanized_mapping = {
    'accha': 'good',
    'acha': 'good',
    'achha': 'good',
    'bahut': 'very',
    'bhut': 'very',
    'kharab': 'bad',
    'bekar': 'bad',
    'sahi': 'correct',
    'nhi': 'not',
    'nahi': 'not',
    'milega': 'receive',
    'mila': 'received',
    'liya': 'take',
    'diya': 'give',
    'lene': 'take',
    'achcha':'good',
    'achhi':'good',
    'achchaa':'good',
    'acchi':'good',
    'aap':'you',
    'bakwaas':'bad',
    'bakwas':'bad',
    'bada':'big'
}

# 3. Download resources and create the final augmented stop_words set
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

STOPWORDS_URL = "https://github.com/manishmaltare/Manish-Maltare/blob/main/english_stopwords.txt"

def load_stopwords_from_github():
    response = requests.get(STOPWORDS_URL)
    if response.status_code == 200:
        stopwords_list = response.text.splitlines()
        return set(stopwords_list)
    else:
        raise Exception(f"Failed to load stopwords from GitHub: {response.status_code}")

english_stopwords = load_stopwords_from_github()
stop_words = english_stopwords.union(set(hindi_noise_words))

### NEW: Add negation words set (minimal but effective list)
negation_words = set(['not', 'no', 'never', 'none', 'neither', 'nor', 'nahi', 'nhi', 'without', 'hardly', 'barely'])  # Expanded but still concise list

# In your code, replace the relevant sections in --- STEP 2: MODIFIED CLEANING FUNCTION ---

### NEW: Expanded list of negation words and phrases
# This includes single words and common 1-3 word phrases. You can add more based on your dataset.
negation_triggers = set([
    'not', 'no', 'never', 'none', 'neither', 'nor', 'nahi', 'nhi',  # Single words
    'is not', 'not good', 'not bad', 'not working',  # 2-word phrases
    'is not good', 'does not work', 'not at all'  # 3-word phrases
])

### NEW: Updated handle_negations function to handle phrases and negate next 1-3 words
def handle_negations(tokens):
    negated = False  # Flag to track if we're in a negated context
    negate_count = 0  # Counter to negate the next 1-3 words
    new_tokens = []   # New list for modified tokens
    i = 0  # Index for iterating through tokens
    while i < len(tokens):
        token = tokens[i]
        
        # Check if the current token or phrase matches a negation trigger
        if token in negation_triggers:
            negated = True  # Start negation
            negate_count = 1  # Start with 1 word to negate, but we can extend
            new_tokens.append(token)  # Keep the negation word as is
        elif negated:
            # Negate the next word(s) - up to 3 words or until the end
            if negate_count <= 3:  # Limit to 3 words
                new_tokens.append(token + "_NEG")  # Modify the word
                negate_count += 1
            else:
                negated = False  # Stop after 3 words
        else:
            new_tokens.append(token)  # Add as is
        
        i += 1  # Move to next token
    
    return new_tokens

def vectorized_clean_series(s: pd.Series):
    s = s.fillna('').astype(str)
    s = s.str.normalize('NFKC')
    s = s.str.lower()

    # 1. Initial cleanup
    s = s.str.replace(r'http\S+|www\.\S+', ' ', regex=True)
    s = s.str.replace(r'\S+@\S+', ' ', regex=True)
    s = s.str.replace(r'<.*?>', ' ', regex=True)
    s = s.str.replace(r'\d+(?:[.,]\d+)*', ' ', regex=True)

    # 2. HINGLISH MAPPING
    def replace_romanized_words(text):
        if not text:
            return ""
        words = text.split()
        words = [romanized_mapping.get(word, word) for word in words]
        return ' '.join(words)
    
    s = s.apply(replace_romanized_words)

    # 3. Final structural cleanup
    s = s.str.replace(r'[^a-z\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    # 4. Tokenize, handle negations, and remove augmented stopwords
    def process_text(x):
        tokens = word_tokenize(x)  # Tokenize
        handled_tokens = handle_negations(tokens)  # Handle negations
        filtered_tokens = [word for word in handled_tokens if word not in stop_words and word]  # Remove stop words
        return ' '.join(filtered_tokens)  # Rejoin
    
    s = s.apply(process_text)  # Use the helper function
    
    return s
# The rest of your code remains unchanged...
# (Continue with df_new_2 processing, model training, and Streamlit app as in your original code)

    
    return s
# Create a new DataFrame with cleaned columns
df_new_2 = df.copy()

df_new_2['title'] = vectorized_clean_series(df_new_2['title'])
df_new_2['body']  = vectorized_clean_series(df_new_2['body'])

df_new_2['rating'] = df['rating']

df_new_2['title_text_length'] = df_new_2['title'].apply(lambda x: len(str(x).split()))

df_new_2['body_text_length'] = df_new_2['body'].apply(lambda x: len(str(x).split()))

# Refined lexicon loader – uses first float after word as score
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

# Load the lexicon from the attached file
sentiment_lexicon = load_lexicon("lexicon.txt")

# Text preprocessing: lowercase, remove punctuation, strip spaces
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# Get sentiment score from the lexicon, default 0
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

# Assuming df_new_2['body'] is your input column
df_new_2['body_sentiments'] = df_new_2['body'].apply(get_sentiment_category)
print(df_new_2[['body', 'body_sentiments']])

df_new_2[['body_sentiments']].value_counts()



rating_map = {
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e',}

df_new_2['rating']=df_new_2['rating'].map(rating_map)
x = df_new_2[['title', 'body','rating']]
y = df_new_2['body_sentiments']



x_train = x.iloc[0:1008]
x_test = x.iloc[1008:]
y_train = y.iloc[0:1008]
y_test = y.iloc[1008:]

x_train['combined'] = x_train['title']+x_train['rating']+x_train['body']
x_test['combined'] = x_test['title']+x_test['rating']+x_test['body']

def preprocess(text):
    if not isinstance(text, str):
        return []
    text = text.lower().translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text.split()
import math
from collections import defaultdict

def build_vocab_idf(corpus):
    vocab = {}
    doc_freq = defaultdict(int)
    processed_docs = []

    for doc in corpus:
        tokens = preprocess(doc)
        processed_docs.append(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1

    vocab = {word: idx for idx, word in enumerate(sorted(doc_freq))}
    total_docs = len(corpus)
    idf = {word: math.log((total_docs + 1) / (doc_freq[word] + 1)) + 1 for word in vocab}

    return vocab, idf, processed_docs


def compute_tfidf(processed_docs, vocab, idf):
    tfidf_matrix = []

    for tokens in processed_docs:
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        total_tokens = len(tokens)
        doc_vector = [0.0] * len(vocab)

        for token in tokens:
            if token in vocab:
                tf_val = tf[token] / total_tokens
                tfidf_val = tf_val * idf[token]
                doc_vector[vocab[token]] = tfidf_val

        tfidf_matrix.append(doc_vector)

    return np.array(tfidf_matrix)

# 1. Create 'combined' column if not already created
x_train['combined'] = x_train['title'] + x_train['rating'] + x_train['body']
x_test['combined'] = x_test['title'] + x_test['rating'] + x_test['body']

# 2. Prepare text data
train_corpus = x_train['combined'].astype(str).tolist()
test_corpus = x_test['combined'].astype(str).tolist()

# 3. Build vocabulary and IDF from training corpus only
vocab, idf, processed_train_docs = build_vocab_idf(train_corpus)

# 4. Vectorize both training and test data
x_train_vectorized = compute_tfidf(processed_train_docs, vocab, idf)

# Process test docs separately using the same vocab and idf
processed_test_docs = [preprocess(doc) for doc in test_corpus]
x_test_vectorized = compute_tfidf(processed_test_docs, vocab, idf)

# 5. Check shapes
print("Train TF-IDF shape:", x_train_vectorized.shape)
print("Test TF-IDF shape:", x_test_vectorized.shape)



import math
from collections import defaultdict

### 1. Preprocessing
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text.split()

### 2. Build vocabulary and calculate IDF
def build_vocab_and_idf(corpus):
    doc_count = len(corpus)
    vocab = {}
    doc_freq = defaultdict(int)
    processed_docs = []

    for doc in corpus:
        tokens = preprocess(doc)
        processed_docs.append(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1

    # Build vocab: word -> index
    vocab = {word: idx for idx, word in enumerate(sorted(doc_freq))}

    # Compute IDF
    idf = {word: math.log(doc_count / (1 + doc_freq[word])) for word in vocab}

    return vocab, idf, processed_docs


def compute_tfidf(processed_docs, vocab, idf):
    tfidf_matrix = []

    for tokens in processed_docs:
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        # Build TF-IDF vector
        doc_vector = [0.0] * len(vocab)
        for token in tokens:
            if token in vocab:
                tf_val = tf[token] / len(tokens)
                tfidf_val = tf_val * idf[token]
                doc_vector[vocab[token]] = tfidf_val
        tfidf_matrix.append(doc_vector)

    return np.array(tfidf_matrix)


train_texts = x_train['combined'].tolist()
test_texts = x_test['combined'].tolist()
all_texts = train_texts + test_texts

# Build vocab & IDF from all data
vocab, idf, all_processed = build_vocab_and_idf(all_texts)

# Split processed docs back into train/test
processed_train = all_processed[:len(train_texts)]
processed_test = all_processed[len(train_texts):]

# Compute TF-IDF vectors
x_train_vectorized = compute_tfidf(processed_train, vocab, idf)
x_test_vectorized = compute_tfidf(processed_test, vocab, idf)



import pickle
from sklearn.svm import SVC

# Assuming you already have x_train_vectorized and y_train

# Train the model (as you've already done)
sv_train = SVC(kernel='linear', gamma='scale', degree=2, C=4.6415888336127775, probability=True)
model_train_tuned = sv_train.fit(x_train_vectorized, y_train)


import string
from collections import defaultdict
import streamlit as st

# Save the trained model to a .pkl file
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model_train_tuned, f)


# Load your trained model
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Load or rebuild your vocab and idf here; they must match the training version
# Example:
# vocab = loaded_vocab
# idf = loaded_idf

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

def compute_tfidf(text_list, vocab, idf):
    tfidf_matrix = []
    for text in text_list:
        tokens = preprocess_text(text)
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        total_tokens = len(tokens)
        doc_vector = np.zeros(len(vocab))
        for token in tokens:
            if token in vocab:
                tf_val = tf[token] / total_tokens
                tfidf_val = tf_val * idf[token]
                doc_vector[vocab[token]] = tfidf_val
        tfidf_matrix.append(doc_vector)
    return np.array(tfidf_matrix)

def predict_sentiment_with_proba(text_list):
    tfidf_feat = compute_tfidf(text_list, vocab, idf)
    probas = model.predict_proba(tfidf_feat)
    class_labels = model.classes_
    return probas, class_labels

# Streamlit app
# At the top of your Streamlit app UI code:
st.title("Sentiment Analysis 😊😐😞")
st.subheader("SVC model - Get sentiment probabilities with confidence levels")

text_input = st.text_area("Enter text (one or more lines)", height=150)

if st.button("Analyze Sentiment"):
    if text_input:
        raw_lines = [line.strip() for line in text_input.split('\n') if line.strip()]
        cleaned_lines = vectorized_clean_series(pd.Series(raw_lines)).tolist()
        probabilities, classes = predict_sentiment_with_proba(cleaned_lines)

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
