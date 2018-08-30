
from graph_tool.all import *
import graph_tool.topology
import csv
import pandas as pd
import numpy as np

#reader = csv.reader(open("sample", 'r'), delimiter=",")
g = load_graph_from_csv("sample")

print (g)


print (g.vertex(1))
print (label_components(g)[1].size)

def nh_subgraph_edge(u,v):
    return set(Gamma(u)).union(Gamma(v))

def GraphFromVertices(vertex_set):
        new_graph = GraphView(g, vfilt=lambda v: v in vertex_set)
        return new_graph

def Gamma(u):
    return set(g.vertex(u).out_neighbors()).union(g.vertex(u).in_neighbors())



def scc_nh(u,v):
    scc = label_components(GraphFromVertices(nh_subgraph_edge(u,v)))[1].size
#    print('computing scc for ',u,v,scc)
    return scc

print (scc_nh(2,12))
