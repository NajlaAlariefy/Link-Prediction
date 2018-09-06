import numpy as np
import pandas as pd
import networkx as nx
import random as random
import matplotlib.pyplot as plt
import math as math
import time as time

# read in the graph through networkx
G =  nx.read_adjlist('/Users/williamrudd/documents/MSc/COMP90051/train.csv', delimiter='\t',create_using=nx.DiGraph() )


# features and neighbourhood functions
def Gamma(u):
    return set(G.successors(u)).union(G.predecessors(u))
def Gamma_in(u):
    return G.predecessors(u)
def Gamma_out(u):
    return G.successors(u)
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
    return len(set(G.successors(u)).intersection(G.predecessors(v)))

# OPPOSITE FRIENDS
def opposite_friends(u,v):
    return int(G.has_edge(v,u))


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
    sum_out = G.out_degree(u)+G.out_degree(v)
    if sum_out == 0:
        return 0
    return 2*len(set(Gamma_out(u)).intersection(set(Gamma_out(v))))/(G.out_degree(u)+G.out_degree(v))


def dice_coef_undirected(u, v):
    sum_out = G.out_degree(u)+G.out_degree(v)
    if sum_out == 0:
        return 0
    return 2*len(set(Gamma(u)).intersection(set(Gamma(v))))/(G.out_degree(u)+G.out_degree(v))



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
    return a/len(out_set)  

def jacard_inneighbours(u,v):
    in_set = Gamma_in(v)
    a = 0
    if not in_set:
        return a
    for i in in_set:
        a = a + jacard_coef(u,i)
    return a/len(in_set)
#def jacard_outneighbors:

    

# Gives the sorted list of nodes and their out degrees
degreelist = G.out_degree()
sorted_d = sorted((value, key) for (key,value) in degreelist.items())
sources_degrees = [(value,key) for (value,key) in sorted_d if value >= 1]
sources = [(key) for (value,key) in sorted_d if value >= 1]



##### A SECOND APPROACH to generating training set
## So, the main problem with the test set is that we seem to be overclassifying
# edges. This means that we are in a sense being "tricked" => our fake edges on
# the training set are too obvious, and our algorithm lacks any capacity to 
# detect edges on the training set.

# How I split this up can be changed, basically for the fake edges, I will 
# generate a real set of sources and a fake set of sources.
# For each of the fake sources, I will generate a target from their 
# neighbourhood that is not in the neighbourhood of the corresponding real source,
# and the edge between these two will be the fake edge.


###G.add_edges_from(edges)

t = time.time()
n = 8000

# NOTE: I am not doing any replacement here, so there are likely to be 
# some duplicates.


# building the set of sources. This is 1.5 times the set that we want,
# because I also add the "dummy" sources for fake edge generation
testsources_list = np.random.choice(sources,int(n+n/2))

# these will be the edges with class 1
testsources_real = testsources_list[0:int(n/2)]

# these will be the sources of the fake edges
testsources_fake_1 = testsources_list[int(n/2):int(n)]

# I use these to generate the targets of the fake edges
testsources_fake_2 = testsources_list[int(n):int(n+n/2)]

# Just formatting the sources into a list from an array
testsources_1 = np.ndarray.tolist(testsources_real)
testsources_2 = np.ndarray.tolist(testsources_fake_1)
testsources = testsources_1 + testsources_2



# initialize real and fake targets.
targets_real = [0]*int(n/2)
targets_fake = [0]*int(n/2)

# loop to generate sources
for i in range(0,int(n/2)):
    q = 0
    # real sources are chosen uniformly from the successor sets
    targets_real[i] = np.random.choice(Gamma_out(testsources_real[i]))
    while q == 0:
        targets_fake[i] = np.random.choice(Gamma_out(testsources_fake_2[i]))
        
        # I think there is a small issue somewhere in this code below.
        # basically, what if the successor set of the node I'm generating from
        # is a subset of the successor set of the one I'm attaching it to?
        # Then we get stuck in the loop forever! 
        
        # my solution is to just break the loop regardless - it seems the 
        # probability of the edge being real is really low so the split ends up
        # being 50.1% to 49.9%, which isn't a big deal, but maybe we can improve
        
        
        # My other workaround is maybe to just break the loop after it iterates
        # over 10 times, maybe.
        if G.has_edge(testsources_fake_1[i],targets_fake[i]):
            q = 1
        else: q = 1
print(time.time()-t)

# concatenate targets into one list.
targets = targets_real + targets_fake



# This appends everything into a data frame
targetarray = np.array(targets)
targetarray
sourced = pd.DataFrame(testsources, columns = ['source'])
sourced['target'] = targetarray

t = time.time()
sourced['class'] = sourced[['source','target']].apply(lambda x: int(G.has_edge(str(x['source']),str(x['target']))),axis=1)
print(time.time() - t)
sourced

edges = list(zip(testsources,targets))
n_before = nx.number_of_edges(G)
G.remove_edges_from(edges)
n_after = nx.number_of_edges(G)

sourced
# FEATURE COMPUTATION #

######## THIS IS WHERE YOU CAN ADD MORE FEATURES ########
######## THIS IS WHERE YOU CAN ADD MORE FEATURES ########
######## THIS IS WHERE YOU CAN ADD MORE FEATURES ########
######## THIS IS WHERE YOU CAN ADD MORE FEATURES ########
######## THIS IS WHERE YOU CAN ADD MORE FEATURES ########


# THESE FEATURES WERE COMPUTED VERY FAST. The longest feature on the 40k size
# data set took 2 minutes on my machine.


######## USE THIS SYNTAX FOR NEW FEATURES #########
######## USE THIS SYNTAX FOR NEW FEATURES #########
######## USE THIS SYNTAX FOR NEW FEATURES #########
######## USE THIS SYNTAX FOR NEW FEATURES #########
######## USE THIS SYNTAX FOR NEW FEATURES #########

t = time.time()
sourced['common_friends'] = sourced[['source','target']].apply(lambda x: common_friends(str(x['source']),str(x['target'])),axis=1)
print('cf',time.time() - t)

t = time.time()
sourced['total_friends'] = sourced[['source','target']].apply(lambda x: total_friends(str(x['source']),str(x['target'])),axis=1)
print('tf',time.time() - t)

t = time.time()
sourced['pref_attach'] = sourced[['source','target']].apply(lambda x: pref_attach(str(x['source']),str(x['target'])),axis=1)
print('pa',time.time() - t)

t = time.time()
sourced['jacard_coef'] = sourced[['source','target']].apply(lambda x: jacard_coef(str(x['source']),str(x['target'])),axis=1)
print('jc',time.time() - t)

t = time.time()
sourced['transitive_friends'] = sourced[['source','target']].apply(lambda x: transitive_friends(str(x['source']),str(x['target'])),axis=1)
print('tf',time.time() - t)

t = time.time()
sourced['opposite_friends'] = sourced[['source','target']].apply(lambda x: opposite_friends(str(x['source']),str(x['target'])),axis=1)
print('of',time.time() - t)

t = time.time()
sourced['adar'] = sourced[['source','target']].apply(lambda x: adar(str(x['source']),str(x['target'])),axis=1)
print('ad',time.time() - t)

t = time.time()
sourced['dice_coef_directed'] = sourced[['source','target']].apply(lambda x: dice_coef_directed(str(x['source']),str(x['target'])),axis=1)
print('dd',time.time() - t)

t = time.time()
sourced['dice_coef_undirected'] = sourced[['source','target']].apply(lambda x: dice_coef_undirected(str(x['source']),str(x['target'])),axis=1)
print('du',time.time() - t)

t = time.time()
sourced['friends_closeness'] = sourced[['source','target']].apply(lambda x: friends_closeness(str(x['source']),str(x['target'])),axis=1)
print('fc',time.time() - t)

t = time.time()
sourced['jacard_outneighbours'] = sourced[['source','target']].apply(lambda x: jacard_outneighbours(str(x['source']),str(x['target'])),axis=1)
print('jo',time.time() - t)

t = time.time()
sourced['jacard_inneighbours'] = sourced[['source','target']].apply(lambda x: jacard_inneighbours(str(x['source']),str(x['target'])),axis=1)
print('ji',time.time() - t)


t = time.time()
sourced['degree_source'] = sourced["source"].apply(lambda x: G.degree(str(x)))
sourced['degree_source_in'] = sourced["source"].apply(lambda x: G.in_degree(str(x)))
sourced['degree_source_out'] = sourced["source"].apply(lambda x: G.out_degree(str(x)))
sourced['degree_target'] = sourced["target"].apply(lambda x: G.degree(str(x)))
sourced['degree_target_in'] = sourced["target"].apply(lambda x: G.in_degree(str(x)))
sourced['degree_target_out'] = sourced["target"].apply(lambda x: G.out_degree(str(x)))





## ADDING CLASS LABELS
#t = time.time()
#sourced['class'] = sourced[['source','target']].apply(lambda x: int(G.has_edge(str(x['source']),str(x['target']))),axis=1)
#print(time.time() - t)

# you can see that they are about the same in ratio.
print(sourced['class'].sum())


# here I convert this into a training set, put the features in the correct
# order, and export.
best_train = sourced.drop(['source','target'],axis = 1)
best_train.mean(axis = 0)
best_train

best_train = best_train[['pref_attach','total_friends','common_friends','jacard_coef',
                         'transitive_friends','opposite_friends','adar','dice_coef_directed',
                         'dice_coef_undirected','friends_closeness',
                         'jacard_outneighbours',
                         'jacard_inneighbours','degree_source','degree_source_in','degree_source_out',
                         'degree_target','degree_target_in','degree_target_out','class']]
best_train 
best_train.to_csv('/Users/williamrudd/documents/MSc/COMP90051/train_8k_joji.csv',index = None)


best_train = best_train[['pref_attach','total_friends','common_friends','jacard_coef',
                         'transitive_friends','opposite_friends','adar','degree_source','degree_source_in','degree_source_out',
                         'degree_target','degree_target_in','degree_target_out','class']]
# 

best_train.to_csv('/Users/williamrudd/documents/MSc/COMP90051/train_4k_13features.csv',index = None)






#### CHECKING FOR DUPLICATES and REMOVING THEM
sourced_unique = sourced.drop_duplicates(subset = ['source','target'])
print(sourced_unique['class'].sum())

train_unique = sourced_unique.drop(['source','target'],axis = 1)
train_unique.mean(axis = 0)

train_unique = train_unique[['pref_attach','total_friends','common_friends','degree_source',
         'degree_source_in','degree_source_out','degree_target','degree_target_in','degree_target_out','class']]
train_unique.to_csv('/Users/williamrudd/documents/MSc/COMP90051/train_friday_night_40k_unique.csv',index = None)

# removed duplicates - made little to no difference...










### COMPUTING THE EXACT SAME FEATURES ON THE PUBLIC TEST SET
dft = pd.read_csv('/Users/williamrudd/documents/MSc/COMP90051/test_public.csv',delimiter = '\t')

dft = dft.drop(['Id'],axis = 1)

t = time.time()
dft['pref_attach'] = dft[['Source','Sink']].apply(lambda x: pref_attach(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)


t = time.time()
dft['total_friends'] = dft[['Source','Sink']].apply(lambda x: total_friends(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)

t = time.time()
dft['common_friends'] = dft[['Source','Sink']].apply(lambda x: common_friends(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)


t = time.time()
dft['jacard_coef'] = dft[['Source','Sink']].apply(lambda x: jacard_coef(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)

t = time.time()
dft['transitive_friends'] = dft[['Source','Sink']].apply(lambda x: transitive_friends(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)

t = time.time()
dft['opposite_friends'] = dft[['Source','Sink']].apply(lambda x: opposite_friends(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)

t = time.time()
dft['adar'] = dft[['Source','Sink']].apply(lambda x: adar(str(x['Source']),str(x['Sink'])),axis=1)
print('ad',time.time() - t)

t = time.time()
dft['jacard_outneighbours'] = dft[['Source','Sink']].apply(lambda x: jacard_outneighbours(str(x['Source']),str(x['Sink'])),axis=1)
print('jo',time.time() - t)

t = time.time()
dft['jacard_inneighbours'] = dft[['Source','Sink']].apply(lambda x: jacard_inneighbours(str(x['Source']),str(x['Sink'])),axis=1)
print('ji',time.time() - t)

dft['degree_source'] = dft["Source"].apply(lambda x: G.degree(str(x)))
dft['degree_source_in'] = dft["Source"].apply(lambda x: G.in_degree(str(x)))
dft['degree_source_out'] = dft["Source"].apply(lambda x: G.out_degree(str(x)))
dft['degree_target'] = dft["Sink"].apply(lambda x: G.degree(str(x)))
dft['degree_target_in'] = dft["Sink"].apply(lambda x: G.in_degree(str(x)))
dft['degree_target_out'] = dft["Sink"].apply(lambda x: G.out_degree(str(x)))
dft['class'] = 1

dft = dft.drop(['Source','Sink'],axis = 1)

dft.to_csv('/Users/williamrudd/documents/MSc/COMP90051/test_adarjoji.csv',index = None)









## ignore everything past here





dft = pd.read_csv('/Users/williamrudd/documents/MSc/COMP90051/test_adarjoji.csv')

dfs = pd.read_csv('/Users/williamrudd/documents/MSc/COMP90051/test_public.csv',delimiter = '\t')
dfs = dfs.drop(['Id'],axis = 1)

t = time.time()
dfs['pref_attach'] = dfs[['Source','Sink']].apply(lambda x: pref_attach(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)


t = time.time()
dft['total_friends'] = dfs[['Source','Sink']].apply(lambda x: total_friends(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)

t = time.time()
dfs['common_friends'] = dfs[['Source','Sink']].apply(lambda x: common_friends(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)


t = time.time()
dfs['jacard_coef'] = dfs[['Source','Sink']].apply(lambda x: jacard_coef(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)

t = time.time()
dfs['transitive_friends'] = dfs[['Source','Sink']].apply(lambda x: transitive_friends(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)

t = time.time()
dfs['opposite_friends'] = dfs[['Source','Sink']].apply(lambda x: opposite_friends(str(x['Source']),str(x['Sink'])),axis=1)
print(time.time() - t)


t = time.time()
dfs['adar'] = dft['adar']
print('ad',time.time() - t)


t = time.time()
dfs['dice_coef_directed'] = dfs[['Source','Sink']].apply(lambda x: dice_coef_directed(str(x['Source']),str(x['Sink'])),axis=1)
print('dd',time.time() - t)


t = time.time()
dfs['dice_coef_undirected'] = dfs[['Source','Sink']].apply(lambda x: dice_coef_undirected(str(x['Source']),str(x['Sink'])),axis=1)
print('du',time.time() - t)


t = time.time()
dfs['friends_closeness'] = dfs[['Source','Sink']].apply(lambda x: friends_closeness(str(x['Source']),str(x['Sink'])),axis=1)
print('fc',time.time() - t)

t = time.time()
dfs['jacard_outneighbours'] = dft['jacard_outneighbours']

t = time.time()
dfs['jacard_inneighbours'] = dft['jacard_inneighbours']

dfs['degree_source'] = dfs["Source"].apply(lambda x: G.degree(str(x)))
dfs['degree_source_in'] = dfs["Source"].apply(lambda x: G.in_degree(str(x)))
dfs['degree_source_out'] = dfs["Source"].apply(lambda x: G.out_degree(str(x)))
dfs['degree_target'] = dfs["Sink"].apply(lambda x: G.degree(str(x)))
dfs['degree_target_in'] = dfs["Sink"].apply(lambda x: G.in_degree(str(x)))
dfs['degree_target_out'] = dfs["Sink"].apply(lambda x: G.out_degree(str(x)))
dfs['class'] = 1

dfs = dfs.drop(['Source','Sink'],axis = 1)

dfs
dfs.to_csv('/Users/williamrudd/documents/MSc/COMP90051/test_05sep.csv',index = None)


dft.mean(axis = 0)
best_train.mean(axis = 0)