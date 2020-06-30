import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk.data
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import nltk.classify
import os
from modules import utils
stop_words = stopwords.words('english')
threshold = 0.7
mu = 0.001


# create contextual graph from given document
def c_graph(filename):
    file = open(filename, "r")
    filedata = file.readlines()
    article = filedata[0].split(".")
     
    doc = np.char.lower(article)
    
    cgraph = []
    for node in doc:
        node = node.replace('[^A-Za-z]+',' ')
        node = re.sub(r'[0-9]+', ' ', node)
        node = node.replace(',',' ')
        node = re.sub(r'\d',' ', node)
        node = re.sub(r':',' ', node)
        node = re.sub('-',' ', node)
        node = re.sub(r'\([^)]*\)', ' ', node)
        node = re.sub(r'!@#$;!*%&~^', ' ', node)
        node = ' '.join([w for w in node.split() if w not in stop_words])
        cgraph.append(node)
    cgraph = [i for i in cgraph if i]
    
    cgraph = utils.stem_words(cgraph)
    
    #print(cgraph)
    
    return cgraph
