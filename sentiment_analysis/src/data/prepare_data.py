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
nltk.download('stopwords')

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

df = pd.read_json('../../data/raw/newsgroups.json')

# Convertir a una lista
data = df.content.values.tolist()

# Eliminar emails
data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]

# Eliminar newlines
data = [re.sub(r'\s+', ' ', sent) for sent in data]

# Eliminar comillas
data = [re.sub(r"\'", "", sent) for sent in data]

# Save in plk files
with open(r"../../data/interim/prepared_data.pkl", "wb") as output_file:
    pickle.dump(data, output_file)