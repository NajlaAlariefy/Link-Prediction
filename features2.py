# File for testing feature implementation
import networkx as nx
import time
import pandas as pd
G =  nx.read_adjlist('/Users/williamrudd/documents/MSc/COMP90051/train.txt', delimiter='\t',create_using=nx.DiGraph() )

# Defining intersection and union of lists.
G.degree('20388')
# Trying to be consistent with notation from the paper to avoid confusion.
# Neighborhood functions
def Gamma(u):
    return set(G.successors(u)).union(G.predecessors(u))
def Gamma_in(u):
    return G.predecessors(u)
def Gamma_out(u):
    return G.successors(u)
def Gamma_plus(u):
    return set(Gamma(u)).union({u})    


## maybe not use.
def Gamma_union(u,v):
    return set(Gamma(u)).union(Gamma(v))

def Gamma_union_plus(u,v):
    return set(Gamma_plus(u)).union(Gamma_plus(v))
# DEFINING FEATURES


# FASTEST edge features => total of 4 features.
def total_friends(x):
    return len(set(Gamma(str(x[0]))).union(Gamma(x[1])))
# PREFERENTIAL ATTACHMENT
def pref_attach(u,v):
    return len(Gamma(u))*len(Gamma(v))

# COMMON FRIENDS
def common_friends(u,v):
    return len(set(Gamma(u)).intersection(Gamma(v)))

# JACARD'S COEFFICIENT
def jacard_coef(u,v):
    return common_friends(u,v)/total_friends(u,v)

# TRANSITIVE FRIENDS
def trans_friends(u,v):
    return len(set(G.successors(u)).intersection(G.predecessors(v)))

# OPPOSITE FRIENDS
def opposite_friends(u,v):
    return int(G.has_edge(v,u))

# Vertex features are all very fast and already implemented in networkx
# example:
G.degree('20388')
G.in_degree('20388')
G.out_degree('20388')


# SLOW FEATURES - need to be rewritten in a more efficient manner. To start, I will
# be taking a look at 

def friends_measure(u,v):
    counter = 0
    for i in Gamma(u):
        for j in Gamma(v):
            if (i == j) or (G.has_edge(i,j) == True) or (G.has_edge(j,i) == True):
                counter = counter + 1
    return counter



# TESTING ON THE 2.4mil size subgraph
data = pd.read_csv('/Users/williamrudd/documents/MSc/COMP90051/dataset.csv')
len(data)

# Compute degree features
t = time.time()
data['degree_source'] = data["source"].apply(lambda x: G.degree(str(x)))
t1 = time.time() - t
print(t1)

t = time.time()
data['degree_source_in'] = data["source"].apply(lambda x: G.in_degree(str(x)))
t2 = time.time() - t
print(t2)

t = time.time()
data['degree_source_out'] = data["source"].apply(lambda x: G.out_degree(str(x)))
t3 = time.time() - t
print(t3)

t = time.time()
data['degree_target'] = data["target"].apply(lambda x: G.degree(str(x)))
t4 = time.time() - t
print(t4)

t = time.time()
data['degree_target_in'] = data["target"].apply(lambda x: G.in_degree(str(x)))
t5 = time.time() - t
print(t5)

t = time.time()
data['degree_target_out'] = data["target"].apply(lambda x: G.out_degree(str(x)))
t6 = time.time() - t
print(t6)

# compute edge features



# SUBGRAPH FUNCTIONS/FEATURES
# I represent these in subgraph form rather than edge list, as specified in the
# paper.

# nh subgraph for vertex (unused as of this moment)
def nh_subgraph_vertex(u):
    return G.subgraph(Gamma(u))

def nh_subgraph_vertex_plus(u):
    return G.subgraph(Gamma_plus(u))

# nh subgraph for edge
def nh_subgraph_edge(u,v):
    return G.subgraph(set(Gamma(u)).union(Gamma(v)))

def nh_subgraph_edge_plus(u,v):
    return G.subgraph(set(Gamma_plus(u)).union(Gamma_plus(v)))

def inner_subgraph(u,v):
    g1 = Gamma(u)
    g2 = Gamma(v)
    # some function that grabs a full edgelist
    # for some edge (a,b)
    g = nh_subgraph_edge(u,v)
    e = g.edges()
    inner = nx.DiGraph() # create empty subgraph structure
    for i in e:
        if ((i[0] in g1) and (i[1] in g2)) or ((i[1] in g1) and (i[0] in g2)):
            inner.add_edge(i[0],i[1]) # add this edge to inner
    return inner

# scc nh subgraph
def scc_nh(u,v):
    return nx.number_strongly_connected_components(nh_subgraph_edge(u,v))  

# scc nh subgraph +
def scc_nh_plus(u,v):
    return nx.number_strongly_connected_components(nh_subgraph_edge_plus(u,v)) 

# scc inner subgraph
def scc_inner(u,v):
    return nx.number_strongly_connected_components(inner_subgraph(u,v))


# Related: number of nodes in the above subgraphs
def size_nh(u,v):
    return nx.number_of_nodes(nh_subgraph_edge(u,v))  

# size nh subgraph +
def size_nh_plus(u,v):
    return nx.number_of_nodes(nh_subgraph_edge_plus(u,v)) 

# size inner subgraph
def size_inner(u,v):
    return nx.number_of_nodes(inner_subgraph(u,v))


