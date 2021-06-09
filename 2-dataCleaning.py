from nltk import data
import pandas as pd 
import numpy as np
import os
import pyprind
import matplotlib.pyplot as plt
import re
import itertools
import datetime
import csv
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize, RegexpTokenizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.probability import FreqDist

dataframe = pd.read_csv('reviews.csv')

def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in  string.punctuation])
    return no_punct

tokenizer = RegexpTokenizer(r'\w+')
def remove_stopwords(text):
    stopwordsList = stopwords.words('english')
    stopwordsList.append('dont')
    stopwordsList.append('didnt')
    stopwordsList.append('doesnt')
    stopwordsList.append('cant')
    stopwordsList.append('couldnt')
    stopwordsList.append('couldve')
    stopwordsList.append('im')
    stopwordsList.append('ive')
    stopwordsList.append('isnt')
    stopwordsList.append('theres')
    stopwordsList.append('wasnt')
    stopwordsList.append('wouldnt')
    stopwordsList.append('a')
    stopwordsList.append('also')
    stopwordsList.append('rt')
    stopwordsList.append('br')
    words = [w for w in text if w not in stopwordsList]
    return words

lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
    lem_text = " ".join([lemmatizer.lemmatize(i) for i in text])
    return lem_text

dataframe['processed_text'] = None
tokenizer = RegexpTokenizer(r'\w+')

dataframe.processed_text = dataframe.review.apply(lambda x: remove_punctuation(x), 1)
dataframe.processed_text = dataframe.processed_text.apply(lambda x: tokenizer.tokenize(x.lower()))
dataframe.processed_text = dataframe.processed_text.apply(lambda x: remove_stopwords(x))
dataframe.processed_text = dataframe.processed_text.apply(lambda x: word_lemmatizer(x))
dataframe.processed_text = dataframe.processed_text.apply(lambda elem: re.sub(r"([0-9]+)", "", elem))

# Convert into a single CSV file.
dataframe.to_csv('./clean_reviews.csv', index=False)
print("CVS file is created")