import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sentence_transformers import SentenceTransformer

import spacy
import faiss

def normalize_input(input):
    """ hola
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(input)
    # takes out stop words and only alphanumeric characters
    normalized_list = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(normalized_list)

def load_model(path_embeddings, path_indexes, model = "all-MiniLM-L6-v2"):
    """ hola2
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.load(path_embeddings)
    indexes = faiss.read_index(path_indexes)
    
    return model, embeddings, indexes

def search_recipe(query, model, indexes, df, top_number = 30):
    """ hola3
    
    """
    query_vector = model.encode([query]).astype("float32")
    
    distances, indices = indexes.search(query_vector, top_number)
        
    results = df.iloc[indices[0]]
    return results