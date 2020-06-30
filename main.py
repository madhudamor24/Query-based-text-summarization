import os
import sys
import os.path
import math
import nltk
nltk.download('wordnet')
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from modules import cgraph
from modules import igraph
from modules import ctree
from modules import sgraph
from modules import utils
import documents
import networkx as nx
import re
import scipy as sp
from operator import itemgetter 
import matplotlib.pyplot as plt
stop_words = stopwords.words('english')

def generate_summary():

# take docs input
    file = os.listdir("documents")
    pathname = []
    for i in range(5):
        pathname.append(os.path.join("documents", file[i]))
    
# CGs
    CGs = []
    for filename in pathname:
        CG = cgraph.c_graph(filename)
        CGs.append(CG)
    CGs.sort(key = len, reverse = True)
    print("Length of CG : ",len(CGs))

    
# IG
    IG = igraph.i_graph(CGs)
#     nx.draw(IG)
#     plt.show()
#     plt.title('IG')
#     print("Number of nodes in IG : ",IG.number_of_nodes())
    
    
# query
    query = input('Enter the query : ')
    query = query.strip('.grow')
    #print(query)
    qry = query.split(" ")
    qry = np.char.lower(qry)
    qry = utils.stem_words(qry)

    q = []
    for node in qry:
        node = node.replace('[^A-Za-z]+','')
        node = node.replace(',','')
        node = re.sub(r'\d','', node)
        node = re.sub(r'[0-9]+', '', node)
        node = re.sub('-','', node)
        node = re.sub(r':','', node)
        node = re.sub(r'\([^)]*\)', '', node)
        node = re.sub(r'!@#$;!*%&~^', '', node)
        x = ''.join([w for w in node.split() if w not in stop_words])
        x = x.replace('.','')
        q.append(x)
    q = [i for i in q if i]
    #print(q)
    #print("Number of query terms: ",len(q))
    
    
    
# ctree
    CTs = []
    for i in range(len(q)):
        q_term = q[i]
        CTREES = ctree.c_tree(IG, q_term)
        CTs.append(CTREES)

        
# Sgraph
    Sgraph = sgraph.s_graph(IG, CTs)
    
#     nx.draw(Sgraph)
#     plt.show()     
#     plt.title('Sgraph')

    
# SgraphScore

    n = Sgraph.number_of_nodes()
    print("Number of nodes of Sgraph : ",n)
    
    SGraphScore = utils.total_CtreeScore(q,CTs, IG)
    
    try:
        SGraphScore = SGraphScore/math.sqrt(n)
    except ZeroDivisionError:
        SGraphScore = 0
    print("SgraphScore : ",SGraphScore)
  
    
# summary
    summary = []
    x = Sgraph.nodes()
    for node in x:
        summary.append(node)
    summary.sort(key = id, reverse = False)
    
    lemmatizer = WordNetLemmatizer()
    
    Overall_summary = ""
    for sent in summary:
        word_list = nltk.word_tokenize(sent)
        Overall_summary = Overall_summary + (' '.join([lemmatizer.lemmatize(w) for w in word_list])).capitalize() +'. '
        
    print("Summary : ",Overall_summary)


def main():
    generate_summary()

if __name__=="__main__":
    main()