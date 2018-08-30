import numpy as np
import pandas as pd
import networkx as nx
import random as random


# Gives the sorted list of nodes and their out degrees
degreelist = G.out_degree()
sorted_d = sorted((value, key) for (key,value) in degreelist.items())

# Trimming the list to only include nodes with out degree >= 1
sources_degrees = [(value,key) for (value,key) in sorted_d if value >= 1]
sources = [(key) for (value,key) in sorted_d if value >= 1]
sources

# Just testing how to randomly sample from these. if you run this and compare
# it to test (which is just the degree features of the test set), they are 
# quite similar.
testsources = np.random.choice(sources,10,replace=False)
testsources_list = np.ndarray.tolist(testsources)
testsources_list
sourcearray = np.array(testsources)
sourced = pd.DataFrame(sourcearray, columns = ['source'])
sourced
sourced['degree_source'] = sourced["source"].apply(lambda x: G.degree(str(x)))
sourced['degree_source_in'] = sourced["source"].apply(lambda x: G.in_degree(str(x)))
sourced['degree_source_out'] = sourced["source"].apply(lambda x: G.out_degree(str(x)))
print(sourced.mean(axis = 0))
print(test.mean(axis = 0))


# Here is where I sample the targets (and where things may go wrong...)
testsources = np.random.choice(sources,10000,replace=False)
testsources_list = np.ndarray.tolist(testsources)
targets = []
t = time.time()
for i in testsources_list:
    p = random.uniform(0,1)
    if p < 0.5:
        # with half probability add an existing edge
        next = np.random.choice(Gamma_out(i))
        targets.append(next)
    else:
        # with half probability add a random node, which is likely not an edge.
        # maybe this process should be changed. Maybe p = 0.5 is not reasonable,
        # as our classifier seems to give us ~ 66% positives
        next = random.sample(G.nodes(),1)[0]
        targets.append(next)
print(targets)
print(time.time()-t)




# The remaining code I am just computing features to feed into weka.
targetarray = np.array(targets)
targetarray
sourced = pd.DataFrame(testsources, columns = ['source'])
sourced['target'] = targetarray
int(G.has_edge('1501902','21700'))


t = time.time()
sourced['common_friends'] = sourced[['source','target']].apply(lambda x: common_friends(str(x['source']),str(x['target'])),axis=1)
print(time.time() - t)

t = time.time()
sourced['total_friends'] = sourced[['source','target']].apply(lambda x: total_friends(str(x['source']),str(x['target'])),axis=1)
print(time.time() - t)

t = time.time()
sourced['pref_attach'] = sourced[['source','target']].apply(lambda x: pref_attach(str(x['source']),str(x['target'])),axis=1)
print(time.time() - t)

t = time.time()
sourced['jacard_coef'] = sourced[['source','target']].apply(lambda x: jacard_coef(str(x['source']),str(x['target'])),axis=1)
t10 = time.time() - t

t = time.time()
sourced['transitive_friends'] = sourced[['source','target']].apply(lambda x: transitive_friends(str(x['source']),str(x['target'])),axis=1)
t11 = time.time() - t

t = time.time()
sourced['opposite_friends'] = sourced[['source','target']].apply(lambda x: opposite_friends(str(x['source']),str(x['target'])),axis=1)
t12 = time.time() - t


sourced['degree_source'] = sourced["source"].apply(lambda x: G.degree(str(x)))
sourced['degree_source_in'] = sourced["source"].apply(lambda x: G.in_degree(str(x)))
sourced['degree_source_out'] = sourced["source"].apply(lambda x: G.out_degree(str(x)))
sourced['degree_target'] = sourced["target"].apply(lambda x: G.degree(str(x)))
sourced['degree_target_in'] = sourced["target"].apply(lambda x: G.in_degree(str(x)))
sourced['degree_target_out'] = sourced["target"].apply(lambda x: G.out_degree(str(x)))



t = time.time()
sourced['class'] = sourced[['source','target']].apply(lambda x: int(G.has_edge(str(x['source']),str(x['target']))),axis=1)
print(time.time() - t)
sourced

best_train = sourced.drop(['source','target'],axis = 1)
best_train
best_train.mean(axis = 0)

best_train = best_train[['pref_attach','total_friends','common_friends','jacard_coef','transitive_friends','opposite_friends','degree_source',
         'degree_source_in','degree_source_out','degree_target','degree_target_in','degree_target_out','class']]
best_train
best_train.to_csv('/Users/williamrudd/documents/MSc/COMP90051/best_train2.csv',index = None)

print(best_train.mean(axis = 0))
print(test.mean(axis = 0))
sourced
# The code above is producing my train set, which is a bit closer to the test set
# in many of its features
 


### Can ignore much of this, just data exploration.
# SORTING VALUES OF target in degree
testsort = test.sort_values('degree_target_in')
testbottomhalf = testsort[:1000]
testbottomhalf.mean(axis = 0)
edgelist_test = pd.read_csv('/Users/williamrudd/documents/MSc/COMP90051/test_public.csv',delimiter = '\t')
edgelist_test['dup_source'] = edgelist_test.duplicated(subset = ['Source'])
edgelist_test['dup_sink'] = edgelist_test.duplicated(subset = ['Sink'])
Test1 = edgelist_test['dup_source'].sum()
Test2 = edgelist_test['dup_sink'].sum()

# SORTING VALUES OF source out degree
testsort = test.sort_values('degree_target_out')
testsort
testbottomhalf = testsort.iloc[1000:2000]
testbottomhalf.mean(axis = 0)
testbottomhalf



edgelist_train = df
edgelist_train['dup_source'] = edgelist_train.duplicated(subset = ['source'])
edgelist_train['dup_sink'] = edgelist_train.duplicated(subset = ['target'])
Train1 = edgelist_train['dup_source'].sum()
Train2 = edgelist_train['dup_sink'].sum()




# fixing newtest_9features
newtest = pd.read_csv('/Users/williamrudd/documents/MSc/COMP90051/new_test_9features.csv')
newtest = newtest[['pref_attach','total_friends','common_friends','degree_source',
         'degree_source_in','degree_source_out','degree_target','degree_target_in','degree_target_out','class']]

newtest.to_csv('/Users/williamrudd/documents/MSc/COMP90051/newtest.csv',index = None)