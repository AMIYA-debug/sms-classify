import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
wl=WordNetLemmatizer()
def cleaning_part(message):
    corpus = []
    for i in range(len(message)):
        review = re.sub('[^a-zA-Z]', ' ', message[i])  # fix here
        review = review.lower()
        review = review.split()
        review = [wl.lemmatize(word) for word in review]
        review = ' '.join(review)
        corpus.append(review)
    
    return corpus    



    