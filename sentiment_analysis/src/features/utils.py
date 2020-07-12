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

def sent_to_words(sentences):
    for sentence in sentences:
        # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True elimina la puntuación

# Eliminar stopwords
def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Hacer bigrams
def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

# Hacer trigrams
def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Lematización basada en el modelo de POS de Spacy
def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out