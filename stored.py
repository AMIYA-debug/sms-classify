import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
wl=WordNetLemmatizer()
def cleaning_part(message):
    corpus = []
    
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower()
    review = review.split()
    review = [wl.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)
    
    return corpus    
def separate(sentence):
    return [simple_preprocess(s) for s in sentence]


   



    