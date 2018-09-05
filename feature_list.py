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
