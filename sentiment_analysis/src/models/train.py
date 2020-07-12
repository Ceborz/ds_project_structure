import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import pickle

def train():
    with open(r"../../data/interim/corpus.pkl", "rb") as input_file:
        corpus = pickle.load(input_file)

    with open(r"../../data/interim/id2word.pkl", "rb") as input_file:
        id2word = pickle.load(input_file)

    with open(r"../../data/interim/data_lemmatized.pkl", "rb") as input_file:
        data_lemmatized = pickle.load(input_file)
    
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=20, 
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    doc_lda = lda_model[corpus]
    # Score de coherencia
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    return doc_lda

train()