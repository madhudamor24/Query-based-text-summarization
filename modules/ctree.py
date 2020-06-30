from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from operator import itemgetter, attrgetter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
threshold = 0.7
mu = 0.001
from modules import utils
beta = 1
b = 3

# contruct ctree from IG and query term
def c_tree(IG, q_term):
    CTREES = []
    
    for node in IG: 
        ctree = construct_ctree(node, q_term, IG)
        CTREES.append(ctree)
    
    return CTREES 

# construct ctree for a node and query term
def construct_ctree(node, q_term, IG):
    arr = dict()
    for x in IG.neighbors(node):
        alpha = utils.ctree_score(x,q_term,node,IG)
        f_t = alpha*utils.sim(node,x)
        wt = utils.weight(x, q_term, IG)
        s_t = beta*wt
        prominence_score = f_t + s_t
        arr[x] = prominence_score

    ctree = nx.Graph()
    if(utils.isQtermPresent(node, q_term) == True):
        i = 0
        for x in arr:
            if i<b:
                ctree.add_edge(node,x)
                i+=1
    else:
        found = False
        for x in arr:
            if found == False:
                ctree.add_edge(node,x)
                found = utils.isQtermPresent(x, q_term)
        if found ==False:
            ctree = nx.empty_graph()

#     nx.draw(ctree)
#     plt.show() 
    return ctree
