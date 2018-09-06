import numpy as np
import pandas as pd
import networkx as nx
import random as random
import math as math
import time as time
import builtins



# features and neighbourhood functions
def Gamma(u):
    return set(builtins.G.successors(u)).union(builtins.G.predecessors(u))
def Gamma_in(u):
    return builtins.G.predecessors(u)
def Gamma_out(u):
    return builtins.G.successors(u)
def Gamma_plus(u):
    return set(Gamma(u)).union({u})
def total_friends(u,v):
    return len(set(Gamma(u)).union(Gamma(v)))
def pref_attach(u,v):
    return len(Gamma(u))*len(Gamma(v))
def common_friends(u,v):
    return len(set(Gamma(u)).intersection(Gamma(v)))
def jacard_coef(u,v):
    return common_friends(u,v)/total_friends(u,v)

# TRANSITIVE FRIENDS
def transitive_friends(u,v):
    return len(set(builtins.G.successors(u)).intersection(builtins.G.predecessors(v)))

# OPPOSITE FRIENDS
def opposite_friends(u,v):
    return int(builtins.G.has_edge(v,u))

# DEFINING friends features of neighborhoods, and adar.
def adar(u,v):
    n = set(Gamma(u)).intersection(Gamma(v))
    a = 0
    if not n:
        return a
    for i in n:
        a = a + 1/math.log(len(Gamma(i))+0.0005)
    return a

def dice_coef_directed(u,v):
    sum_out = builtins.G.out_degree(u)+builtins.G.out_degree(v)
    if sum_out == 0:
        return 0
    return 2*len(set(Gamma_out(u)).intersection(set(Gamma_out(v))))/(builtins.G.out_degree(u)+builtins.G.out_degree(v))

def dice_coef_undirected(u, v):
    sum_out = builtins.G.out_degree(u)+builtins.G.out_degree(v)
    if sum_out == 0:
        return 0
    return 2*len(set(Gamma(u)).intersection(set(Gamma(v))))/(builtins.G.out_degree(u)+builtins.G.out_degree(v))

def friends_closeness(u,v):
    try:
        path_length = nx.shortest_path_length(G, source=u, target=v);
        return math.log(total_friends(u, v))/(10*path_length)
    except nx.NetworkXNoPath:
        return 0

def jacard_outneighbours(u,v):
    out_set = Gamma_out(u)
    a = 0
    if not out_set:
        return a
    for i in out_set:
        a = a + jacard_coef(i,v)
    return a/(len(set(out_set)) +0.0001)

def jacard_inneighbours(u,v):
    in_set = Gamma_in(v)
    a = 0
    if not in_set:
        return a
    for i in in_set:
        a = a + jacard_coef(u,i)
    return a/(len(set(in_set)) +0.0001)
