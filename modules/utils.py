import os
import sys
import nltk
nltk.download('punkt')
import urllib.request
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
threshold = 0.7
d = input("Enter value of bias factor : ")
beta = 1


#CG
# stemmer
def stem_words(CG):
    porter = PorterStemmer()
    stemmed_words = []
    for word in CG:
        stemmed_words.append(porter.stem(word))
    return stemmed_words


# CG,IG
# cosine similarity bettween 2 sentences
def sim(text1, text2):
    text1_list = word_tokenize(text1)  
    text2_list = word_tokenize(text2) 
  
    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    t1 =[];t2 =[] 
  
    # remove stop words from string 
    text1_set = {w for w in text1_list if not w in sw}  
    text2_set = {w for w in text2_list if not w in sw} 
  
    # form a set containing keywords of both strings  
    rvector = text1_set.union(text2_set)  
    for w in rvector: 
        if w in text1_set: t1.append(1) 
        else: t1.append(0) 
        if w in text2_set: t2.append(1) 
        else: t2.append(0) 
    c = 0
  
    # cosine formula  
    for i in range(len(rvector)): 
        c+= t1[i]*t2[i]
    try:
        cosine = c / float((sum(t1)*sum(t2))**0.5) 
    except ZeroDivisionError:
        cosine = 0
    return cosine


# whether node presen in arr or not
def ispr(arr,n):
    l = len(arr)
    for i in range(l):
        if n==arr[i]:
            return True
    return False


# compare node wt with threshold
def cmpNode(arr,n,x):
    m = []
    for r in range(len(arr)):
        x = arr[r]
        o = sim(n,x)
        m.append(o>threshold)
    return m


# find whether q_term present in given node or not
def isQtermPresent(node, q_term):
    word = node.split(" ")
    for i in word:
        if i == q_term:
            return True
    return False


# CT
# Ctree score of query term
def ctree_score(x, q_term, node, IG):
    
    score = []
    for n in IG.neighbors(node):
        score.append(sim(n,node))
    score.sort(reverse = True)
    b = score[0]

    wts = []
    for ne in IG.neighbors(x):
        if ne == node:
            continue
        n_w = weight(ne, q_term, IG)
        wts.append(n_w)
    wts.sort(reverse = True)
    
    sum = 0
    for x in wts[:3]:
        sum += x
    a = sum/3
    alpha = (a*1.5)/b
    return alpha


# Weigh of node w.r.t query term
def weight(node, q_term, IG):
    node_wt_q = graph_wt_q(IG, q_term)
    node_wt = neighbour_node_wt(node, q_term, IG)
    
    try:
        first = (sim(node,q_term))/(node_wt_q)
    except ZeroDivisionError:
        first = 0
    second = node_wt
    
    total_wt = float(d)*first + float(1-float(d))*second
    
    return total_wt


# Weight of all graph nodes w.r.t query term
def graph_wt_q(IG, q_term):
    N = IG.nodes()
    node_wt_q = 0
    
    for x in N:
        node_wt_q += sim(x,q_term)

    return node_wt_q


# neighbour's node wts
def neighbour_node_wt(node, q_term, IG):
    node_wt = 0
    for v in IG.neighbors(node):
        adj_node_wt = 0
        for u in IG.neighbors(v):
            adj_node_wt += sim(u,v)
        cdn = sim(node,v)
        if cdn < threshold or IG.degree(v) < 2 :
            return node_wt
        node_wt += (cdn*weight(v, q_term,IG))/(adj_node_wt)
    return node_wt


# ranking 
# total ctree score w.r.t query
def total_CtreeScore(q,CTs, IG):
    tot_score = 0 
    for i in range(len(q)):
        q_term = q[i]
        tot_score += score(q_term, CTs, IG)
    return tot_score

def score(q_term, CTs, IG):
    t_score_q = 0
    for node in IG:
        fst = beta*weight(node, q_term, IG)
        scnd = 0
        for CTREES in CTs:
            for ctree in CTREES:
                if node in ctree:
                    for v in ctree.neighbors(node):
                        if ctree.degree(v) ==1:
                            scnd += ctree_score(v, q_term, node, IG)*sim(node,v)+beta*weight(v, q_term, IG)
        t_score_q += fst + scnd
        
    return t_score_q
