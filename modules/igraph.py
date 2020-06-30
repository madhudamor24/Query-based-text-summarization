from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
threshold = 0.7
mu = 0.001
from modules import utils
import math


# IG using CG's
def i_graph(CGs):
    
    graphs  = []
    for cgraph in CGs:
        CG = nx.Graph()
        for node1 in cgraph:
            for node2 in cgraph:
                if node1==node2:
                    continue
                wt = utils.sim(node1, node2)
                if(wt >= mu):
                    CG.add_edge(node1,node2, weight = wt)
        graphs.append(CG)
    
    
    IG = nx.Graph(graphs[0])
    
    igraph = []
    for node in CGs[0]:
        igraph.append(node)
    
    eta = 0
    for CG in CGs:
        for node in CG:
            eta += 1
             
    node_id = dict()
    nodes_igraph=1
    for node in IG:
        id = math.sqrt((nodes_igraph-1)*eta)
        node_id[node] = id;
        nodes_igraph += 1
    
    
    i = 1
    while i < len(CGs):
        z = CGs[i]
        for j in range(len(CGs[i])):
            n = CGs[i][j]
            y = z
            deg = 0
            for t in range(len(y)):
                node = y[t]
                if n == node:
                    continue
                deg_nb = graphs[i].degree[node]
                if deg_nb>deg:
                    deg = deg_nb
                    p = node             
            
            l1 = []
            for k in range(len(CGs[i])):
                x = CGs[i][k]
                if not(utils.ispr(igraph,x)):
                    l1.append(x)
        
            l2 = []
            for k in range(len(CGs[i])):
                x = CGs[i][k]
                if x not in graphs[i].neighbors(p):
                    l2.append(x)
            
            if utils.ispr(igraph,p):
                if utils.cmpNode(l1,n,x)== [False]:
                    for a in range(len(igraph)):
                        y = igraph[a]
                        if utils.sim(n,y) > mu:
                            IG.add_edge(n, y,weight = utils.sim(n,y))
                            igraph.append(y)
                            node_id[n] = math.sqrt((nodes_igraph-1)*eta)
                            nodes_igraph += 1
            elif utils.cmpNode(l2,n,x)== [False]:
                for a in range(len(igraph)):
                    y = igraph[a]
                    if utils.sim(n,y) > mu:
                        IG.add_edge(n, y,weight = utils.sim(n,y))
                        igraph.append(y)
                        node_id[n] = math.sqrt((nodes_igraph-1)*eta)
                        nodes_igraph += 1
        
        
        i+=1
    
    #nx.draw(IG)
    #plt.show()
    #plt.title('IG')
    return IG
   