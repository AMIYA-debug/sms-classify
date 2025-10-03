import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess

wl = WordNetLemmatizer()

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

def avg_word2vec(doc):
    return np.mean(doc, axis=0)  

def conversion(tokenized_input):
    loaded_model = Word2Vec.load("vector.model")
    
    word_store = []
    for tokens in tokenized_input:  
        temp = [loaded_model.wv[word] for word in tokens if word in loaded_model.wv]
        word_store.append(temp)

    x = []    
    for temp in word_store:
        if len(temp) == 0:
            x.append(np.zeros(loaded_model.vector_size))  
        else:
            x.append(avg_word2vec(temp)) 
              
    return x

# demo = "congratulations you won"
# first = cleaning_part(demo)        
# second = separate(first)            
# third = conversion(second)          

# print(third)





   



    