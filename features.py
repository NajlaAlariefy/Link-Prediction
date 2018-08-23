# File for testing feature implementation
import networkx as nx

G =  nx.read_adjlist('/Users/williamrudd/documents/MSc/COMP90051/train.txt', delimiter='\t',create_using=nx.DiGraph() )

# Defining intersection and union of lists.
def intersection(a, b):
    return list(set(a) & set(b))
def union(a, b):
    return list(set(a) | set(b))

# Trying to be consistent with notation from the paper to avoid confusion.
# Neighborhood functions
def Gamma(u):
    return union(G.successors(u),G.predecessors(u))
def Gamma_in(u):
    return G.predecessors(u)
def Gamma_out(u):
    return G.successors(u)
def Gamma_plus(u):
    return union(Gamma(u),{u})    


# SUBGRAPH FUNCTIONS
# I represent these in subgraph form rather than edge list, as specified in the
# paper. I think it will be more efficient this way, as we can use the nx.compose
# function and others.

# nh subgraph for vertex
def nh_subgraph_vertex(u):
    return G.subgraph(Gamma(u))

def nh_subgraph_vertex_plus(u):
    return G.subgraph(Gamma_plus(u))

# nh subgraph for edge
def nh_subgraph_edge(u,v):
    return G.subgraph(union(Gamma(u),Gamma(v)))

def nh_subgraph_edge_plus(u,v):
    return G.subgraph(union(Gamma(u),Gamma(v)))

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


# DEFINING FEATURES


# COMPUTATIONALLY FAST
# TOTAL FRIENDS
def total_friends(u,v):
    return len(union(Gamma(u),Gamma(v)))

# PREFERENTIAL ATTACHMENT
def pref_attach(u,v):
    return len(Gamma(u))*len(Gamma(v))

# COMMON FRIENDS
def common_friends(u,v):
    return len(intersection(Gamma(u),Gamma(v)))

# JACARD'S COEFFICIENT
def jacard_coef(u,v):
    return common_friends(u,v)/total_friends(u,v)

# FRIENDS MEASURE (slightly slower)
def friends_measure(u,v):
    counter = 0
    for i in Gamma(u):
        for j in Gamma(v):
            if (i == j) or (G.has_edge(i,j) == True) or (G.has_edge(j,i) == True):
                counter = counter + 1
    return counter


# COMPUTATIONALLY SLOW.
# scc nh subgraph
def scc_nh(u,v):
    return nx.number_strongly_connected_components(nh_subgraph_edge(u,v))  

# scc nh subgraph +
def scc_nh_plus(u,v):
    return nx.number_strongly_connected_components(nh_subgraph_edge_plus(u,v)) 

# scc inner subgraph
def scc_inner(u,v):
    return nx.number_strongly_connected_components(inner_subgraph(u,v))

