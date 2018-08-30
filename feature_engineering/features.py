# File for testing feature implementation
import networkx as nx

G =  nx.read_adjlist('data/train.txt', delimiter='\t',create_using=nx.DiGraph() )

# Defining intersection and union of lists.
def intersection(a, b):
    return list(set(a) & set(b))
def union(a, b):
    return list(set(a) | set(b))

# Trying to be consistent with notation from the paper to avoid confusion.
# Neighborhood functions
def Gamma(G, u):
    return union(G.successors(u),G.predecessors(u))
def Gamma_in(u):
    return G.predecessors(u)
def Gamma_out(u):
    return G.successors(u)
def Gamma_plus(u):
    return union(Gamma(G, u),{u})


# SUBGRAPH FUNCTIONS
# I represent these in subgraph form rather than edge list, as specified in the
# paper. I think it will be more efficient this way, as we can use the nx.compose
# function and others.

# nh subgraph for vertex
def nh_subgraph_vertex(u):
    return G.subgraph(Gamma(G, u))

def nh_subgraph_vertex_plus(u):
    return G.subgraph(Gamma_plus(u))

# nh subgraph for edge
def nh_subgraph_edge(G, u,v):
    return G.subgraph(union(Gamma(G, u),Gamma(G, v)))

def nh_subgraph_edge_plus(G, u,v):
    return G.subgraph(union(Gamma(G, u),Gamma(G, v)))

def inner_subgraph(G, u,v):
    g1 = Gamma(G, u)
    g2 = Gamma(G, v)
    # some function that grabs a full edgelist
    # for some edge (a,b)
    g = nh_subgraph_edge(G, u,v)
    e = g.edges()
    inner = nx.DiGraph() # create empty subgraph structure
    for i in e:
        if ((i[0] in g1) and (i[1] in g2)) or ((i[1] in g1) and (i[0] in g2)):
            inner.add_edge(i[0],i[1]) # add this edge to inner
    return inner


# DEFINING FEATURES


# COMPUTATIONALLY FAST
# TOTAL FRIENDS
def total_friends(G, u, v):
    return len(union(Gamma(G, u),Gamma(G, v)))

# PREFERENTIAL ATTACHMENT
def pref_attach(G, u, v):
    return len(Gamma(G, u))*len(Gamma(G, v))

# COMMON FRIENDS
def common_friends(G, u, v):
    return len(intersection(Gamma(G, u),Gamma(G, v)))

# JACARD'S COEFFICIENT
def jacard_coef(G, u, v):
    return common_friends(G, u, v)/total_friends(G, u, v)

# FRIENDS MEASURE (slightly slower)
def friends_measure(G, u,v):
    counter = 0
    for i in Gamma(G, u):
        for j in Gamma(G, v):
            if (i == j) or (G.has_edge(i,j) == True) or (G.has_edge(j,i) == True):
                counter = counter + 1
    return counter


# COMPUTATIONALLY SLOW.
# scc nh subgraph
def scc_nh(G, u,v):
    return nx.number_strongly_connected_components(nh_subgraph_edge(G, u,v))

# scc nh subgraph +
def scc_nh_plus(G, u,v):
    return nx.number_strongly_connected_components(nh_subgraph_edge_plus(G, u,v))

# scc inner subgraph
def scc_inner(G, u,v):
    return nx.number_strongly_connected_components(inner_subgraph(G, u,v))
