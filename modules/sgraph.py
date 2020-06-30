import networkx as nx
import matplotlib.pyplot as plt

# create sgraph bt merging ctree            
def s_graph(IG, CTs):
    Sgraph = nx.Graph()
    for i in range(len(CTs)):
        CTREES = CTs[i]
        for ctree in CTREES:
            Sgraph = nx.compose(ctree,Sgraph)
        
    return Sgraph