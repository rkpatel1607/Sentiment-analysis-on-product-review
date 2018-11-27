"""
   Topic : Sentiment analysis using product review data
   Authors: Riddhi Patel
"""

import json
import nltk
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_f1
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, naive_bayes, metrics
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

# from nltk import TaggerI
# from nltk.classify.maxent import MaxentClassifier
# from nltk.corpus import treebank
import warnings
warnings.filterwarnings('ignore')

### ---------------------------------------------------- Global -------

# Global lists and Dictionaries
Global_dict = {}
Sentiment_Score = {}
Negative_Pre = ['no', 'not', 'dont', 'don\'t', 'doesnt', 'doesn\'t', 'nothing', 'non', '']
NOV = []    # Negation of Adjectives
NOA = []    # Negation of Adverbs

# defining lists of Tokens which are generated based on Tokens names from NLTK Word-Tokens
POS_arr = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
VERB = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']    # Verbs
ADJ = ['JJ', 'JJR', 'JJS']  # Adjectives
ADV = ['RB', 'RBR', 'RBS']  # Adverbs

stopwords = \
    ['b', 'c' 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
     'x', 'y', 'z', '@', '#', '..', '.', '`', '~', '!', '$', '%', '^', '*', '(', ')', '-', '+', '/"', '//',
     '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '``', ';']

Count_1 = 0
Count_2 = 0
Count_3 = 0
Count_4 = 0
Count_5 = 0
