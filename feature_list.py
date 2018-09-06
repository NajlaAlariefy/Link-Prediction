'''
file: feature_list
description: holds all engineered features for the Link Prediction dataset
'''
import numpy as np
import pandas as pd
import networkx as nx
import random as random
import matplotlib.pyplot as plt
import time
import datetime
from ast import literal_eval
from sklearn.utils import shuffle
from functools import lru_cache
import math




@lru_cache(maxsize=None)
### Node Features
def Gamma(G,u):
    s = gamma_dict.get(u)
    if s is None:
        s = set(G.successors(u)).union(G.predecessors(u))
        gamma_dict[u]=s
    return s
def Gamma_in(u):
    return G.predecessors(u)
def Gamma_out(u):
    return G.successors(u)
def Gamma_plus(u):
    return set(Gamma(u)).union({u})



### Edge Features
def total_friends(G,u,v):
    return len(set(Gamma(G,u)).union(Gamma(G,v)))
def pref_attach(G,u,v):
    return len(Gamma(G,u))*len(Gamma(G,v))
def common_friends(G,u,v):
    return len(set(Gamma(G,u)).intersection(Gamma(G,v)))
def jacard_coef(G,u,v):
    return common_friends(G,u,v)/total_friends(G,u,v)
def pref_attach(G,u,v):
    return len(Gamma(G,u))*len(Gamma(G,v))





### Neighborhood Features
def pref_outneighbours(u):
    out_set = G.successors(u[0])
    res =  [ pref_attach(G,i,u[1]) for i in out_set]
    a = 0 if len(res) == 0 else sum(res)
    return a/(len(set(out_set)) +0.0005)

def pref_inneighbours(u):
    in_set = G.predecessors(u[1])
    res = [pref_attach(G,u[0],i) for i in in_set]
    a = 0 if len(res) == 0 else sum(res)
    return a/(len(set(in_set)) +0.0005)

def jacard_outneighbours(u):
    out_set = Gamma_out(u[0])
    res = [jacard_coef(i,u[1]) for i in out_set]
    a = 0 if len(res) == 0 else sum(res)
    return a/(len(set(out_set)) +0.0005)

def jacard_inneighbours(u):
    in_set = Gamma_in(u[1])
    a = 0
    res = [jacard_coef(u[0],i) for i in in_set]
    a = 0 if len(res) == 0 else sum(res)
    return a/(len(set(in_set)) +0.0005)

def friends_closeness(u):
    try:
        path_length = nx.shortest_path_length(G, source=u[0], target=u[1]);
        return math.log(total_friends(G,u[0], u[1]))/(10*path_length)
    except nx.NetworkXNoPath:
        return 0

def closeness_centrality(u):
    return nx.closeness_centrality(G, u[0])*nx.closeness_centrality(G, u[1])




# Numpy vectorised computation
train = pd.read_csv('train_ref_050918.csv')
vectorised = np.array([train.source,train.target])
vectorised = vectorised.T

# removing edges
sources =train.source[train['class'] == 1]
targets = train.target[train['class'] == 1]
edges = list(zip(sources,targets))
n_before = nx.number_of_edges(G)
G.remove_edges_from(edges)
n_after = nx.number_of_edges(G)


t = time.time()
jacard_inneighbours_f = np.apply_along_axis(jacard_inneighbours, 1, vectorised)
print((time.time() - t))

t = time.time()
jacard_outneighbours_f = np.apply_along_axis(jacard_outneighbours, 1, vectorised)
print((time.time() - t))

t = time.time()
pref_outneighbours_f = np.apply_along_axis(pref_outneighbours, 1, vectorised)
print((time.time() - t))

t = time.time()
pref_inneighbours_f = np.apply_along_axis(pref_inneighbours, 1, vectorised)
print((time.time() - t))


t = time.time()
friends_closeness_f = np.apply_along_axis(friends_closeness, 1, vectorised)
print((time.time() - t))

train['jacard_inneighbours'] = jacard_inneighbourss
train['jacard_outneighbours'] = jacard_outneighbourss
train['friends_closeness'] = friends_closeness_f
train['pref_outneighbours'] = pref_outneighbours_f
train['pref_inneighbours'] = pref_inneighbours

train.to_csv('train.csv', index=None)
