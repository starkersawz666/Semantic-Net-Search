_ = """
This file contains the code for computing text vector representations using word2vec model.
Run this file to generate a new dataset with text vector representations once you change the word2vec model or the dataset.
"""

import gensim.downloader as api
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import yaml

def load_config():
    file_path = '../config/config.yaml'
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# download and load word2vec model
for resource in config['nltk_resources']:
    nltk.download(resource)
model = api.load(config['word2vec_model']['type'])

file_path = "../" + config['datasets']['without_vectors']
df = pd.read_csv(file_path)
df['merged_text'] = ''
for index, row in df.iterrows():
    merged_text = ''
    text_cols = ['name', 'description', 'category', 'node_meaning', 'edge_meaning']
    for col in text_cols:
        if not pd.isnull(row[col]):
            merged_text += row[col] + '; '
    if row['bipartite'] == True:
        merged_text += 'bipartite; '
    elif row['bipartite'] == False:
        merged_text += 'unipartite; '
    if row['directed'] == True:
        merged_text += 'directed; '
    elif row['directed'] == False:
        merged_text += 'undirected; '
    if row['weighted'] == True:
        merged_text += 'weighted; '
    elif row['weighted'] == False:
        merged_text += 'unweighted; '
    if row['positive_weights'] == True:
        merged_text += 'positive weights; '
    if row['negative_weights'] == True:
        merged_text += 'negative weights; '
    if row['multiple_edges'] == True:
        merged_text +='multiple edges; '
    elif row['multiple_edges'] == False:
        merged_text +='single edges; '
    if row['reciprocal'] == True:
        merged_text +='reciprocal; '
    elif row['reciprocal'] == False:
        merged_text += 'non-reciprocal; '
    if row['directed_cycle'] == True:
        merged_text += 'contains directed cycle; '
    elif row['directed_cycle'] == False:
        merged_text += 'no directed cycle; '
    if row['loops'] == True:
        merged_text += 'contains loops; '
    elif row['loops'] == False:
        merged_text += 'no loops; '
    if row['timestamp'] == True:
        merged_text += 'includes timestamp; '
    elif row['timestamp'] == False:
        merged_text += 'no timestamp; '

    df.at[index,'merged_text'] = merged_text

# Text preprocessing function, including tokenization and stop word removal
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    stop_words = set(stopwords.words('english'))  # Stop words
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Compute text similarity function
def compute_similarity(text1, text2):
    # Preprocess text
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)

    # Compute text vector representations
    vector1 = sum(model.get_vector(token) for token in tokens1 if token in model.key_to_index)
    vector2 = sum(model.get_vector(token) for token in tokens2 if token in model.key_to_index)
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray or len(vector1) != len(vector2):
        return -1
    # Compute cosine similarity
    similarity_score = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    return similarity_score

cnt = 0
df['vector'] = None
df['index'] = None
for index, row in df.iterrows():
    tokens = preprocess_text(row['merged_text'])
    vector = sum(model.get_vector(token) for token in tokens if token in model.key_to_index)
    df.at[index,'vector'] = vector
    df.at[index, 'index'] = cnt
    cnt += 1

df.to_csv("../" + config['datasets']['with_vectors'])