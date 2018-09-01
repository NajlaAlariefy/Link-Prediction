import numpy as np
import pandas as pd
import networkx as nx
#import random as random
#import matplotlib.pyplot as plt
import time

train_file_path = 'C:/Users/Modarres/Desktop/train_n.txt'

G = nx.read_adjlist(train_file_path, delimiter='\t', create_using=nx.DiGraph(), nodetype=int)
gamma_dict = dict()

print("graph was read successfully")

def Gamma(u):
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





# Gives the sorted list of nodes and their out degrees
degreelist = G.out_degree()
degreelist = list(degreelist)
sorted_d = sorted((value, key) for [key,value] in degreelist)
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
n = 40000

# NOTE: I am not doing any replacement here, so there are likely to be
# some duplicates.


# building the set of sources. This is 1.5 times the set that we want,
# because I also add the "dummy" sources for fake edge generation
testsources_list = np.random.choice(sources,int(n+n/2))

# these will be the edges with class 1 (1:20000)
testsources_real = testsources_list[0:int(n/2)]

# these will be the sources of the fake edges (20000:40000)
testsources_fake_1 = testsources_list[int(n/2):int(n)]

# I use these to generate the targets of the fake edges (40000:60000)
testsources_fake_2 = testsources_list[int(n):int(n+n/2)]

# Just formatting the sources into a list from an array
testsources_1 = np.ndarray.tolist(testsources_real)
testsources_2 = np.ndarray.tolist(testsources_fake_1)
testsources = testsources_1 + testsources_2



# initialize real and fake targets.
targets_real = np.zeros(int(n/2)) #[0]*int(n/2)
targets_fake = np.zeros(int(n/2)) #[0]*int(n/2)
t = time.time()

# loop to generate sources


for i in range(0,int(n/2)):
    q = 0
    # real sources are chosen uniformly from the successor sets
    targets_real[i] = np.random.choice(list(Gamma_out(testsources_real[i])))
    while q == 0:
        targets_fake[i] = np.random.choice(list(Gamma_out(testsources_fake_2[i])))

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
targets = np.append(targets_real,targets_fake)


# This appends everything into a data frame
targetarray = np.array(targets)
targetarray
sourced = pd.DataFrame(testsources, columns = ['source'])
sourced['target'] = targetarray




# FEATURE COMPUTATION #

######## THIS IS WHERE YOU CAN ADD MORE FEATURES ########
######## THIS IS WHERE YOU CAN ADD MORE FEATURES ########


# dice similarity - ref https://opus.lib.uts.edu.au/bitstream/10453/43302/1/final.pdf
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

# closeness measure
def closeness(u,v):
    try:
        path_length = nx.shortest_path_length(G, source=u, target=v);
        return path_length
    except nx.NetworkXNoPath:
        return 100000

def closeness_centrality(u,v):
    return nx.closeness_centrality(G, u)*nx.closeness_centrality(G, v)




# THESE FEATURES WERE COMPUTED VERY FAST. The longest feature on the 40k size
# data set took 2 minutes on my machine.


######## USE THIS SYNTAX FOR NEW FEATURES #########
######## USE THIS SYNTAX FOR NEW FEATURES #########
######## USE THIS SYNTAX FOR NEW FEATURES #########
######## USE THIS SYNTAX FOR NEW FEATURES #########
######## USE THIS SYNTAX FOR NEW FEATURES #########

t = time.time()
sourced['common_friends'] = sourced[['source','target']].apply(lambda x: common_friends(str(x['source']),str(x['target'])),axis=1)
print(time.time() - t)

t = time.time()
sourced['total_friends'] = sourced[['source','target']].apply(lambda x: total_friends(x['source'],x['target']),axis=1)
print(time.time() - t)

t = time.time()
sourced['pref_attach'] = sourced[['source','target']].apply(lambda x: pref_attach(x['source'],x['target']),axis=1)
print(time.time() - t)

t = time.time()
sourced['jacard_coef'] = sourced[['source','target']].apply(lambda x: jacard_coef(x['source'],x['target']),axis=1)
print(time.time() - t)

t = time.time()
sourced['transitive_friends'] = sourced[['source','target']].apply(lambda x: transitive_friends(x['source'],x['target']),axis=1)
print(time.time() - t)

t = time.time()
sourced['opposite_friends'] = sourced[['source','target']].apply(lambda x: opposite_friends(x['source'],x['target']),axis=1)
print(time.time() - t)

sourced['degree_source'] = sourced["source"].apply(lambda x: G.degree(x))
sourced['degree_source_in'] = sourced["source"].apply(lambda x: G.in_degree(x))
sourced['degree_source_out'] = sourced["source"].apply(lambda x: G.out_degree(x))
sourced['degree_target'] = sourced["target"].apply(lambda x: G.degree(x))
sourced['degree_target_in'] = sourced["target"].apply(lambda x: G.in_degree(x))
sourced['degree_target_out'] = sourced["target"].apply(lambda x: G.out_degree(x))





## ADDING CLASS LABELS
t = time.time()
sourced['class'] = sourced[['source','target']].apply(lambda x: int(G.has_edge(x['source'],x['target'])),axis=1)
print(time.time() - t)

# you can see that they are about the same in ratio.
print(sourced['class'].sum())


# here I convert this into a training set, put the features in the correct
# order, and export.
best_train = sourced.drop(['source','target'],axis = 1)
best_train.mean(axis = 0)

best_train = best_train[['pref_attach','total_friends','common_friends','jacard_coef','transitive_friends','opposite_friends','degree_source',
         'degree_source_in','degree_source_out','degree_target','degree_target_in','degree_target_out','class']]
best_train
best_train.to_csv('train_friday_night_40k_12features.csv',index = None)




#### CHECKING FOR DUPLICATES and REMOVING THEM
sourced_unique = sourced.drop_duplicates(subset = ['source','target'])
print(sourced_unique['class'].sum())

train_unique = sourced_unique.drop(['source','target'],axis = 1)
train_unique.mean(axis = 0)

train_unique = train_unique[['pref_attach','total_friends','common_friends','degree_source',
         'degree_source_in','degree_source_out','degree_target','degree_target_in','degree_target_out','class']]
train_unique.to_csv('train_friday_night_40k_unique.csv',index = None)

# removed duplicates - made little to no difference...
