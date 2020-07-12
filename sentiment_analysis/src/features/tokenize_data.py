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

import nltk
from utils import sent_to_words
from utils import remove_stopwords
from utils import make_bigrams
from utils import make_trigrams
from utils import lemmatization

nltk.download('stopwords')

# Load data
data = None
with open(r"../../data/interim/prepared_data.pkl", "rb") as input_file:
    data = pickle.load(input_file)

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

data_words = list(sent_to_words(data))

# Construimos modelos de bigrams y trigrams
# https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Aplicamos el conjunto de bigrams/trigrams a nuestros documentos
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Eliminamos stopwords
data_words_nostops = remove_stopwords(data_words, stop_words)
# Formamos bigrams
data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
# python3 -m spacy download en_core_web_lg
# Inicializamos el modelo 'en_core_web_lg' con las componentes de POS únicamente
nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

# Lematizamos preservando únicamente noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Creamos diccionario
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Save in plk files
with open(r"../../data/interim/corpus.pkl", "wb") as output_file:
    pickle.dump(corpus, output_file)
with open(r"../../data/interim/id2word.pkl", "wb") as output_file:
    pickle.dump(id2word, output_file)
