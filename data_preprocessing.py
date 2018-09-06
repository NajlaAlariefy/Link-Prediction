import numpy as np
import pandas as pd
import networkx as nx
import random as random
import matplotlib.pyplot as plt
import math as math
import time as time
from feature_list import Gamma_out

def generate_data(G, n_samples=40000):

    # NOTE: No sample replacement is done here, so there are likely to be
    # some duplicates.
    print('Sampleing without replacement. N =', n_samples)

    # Gives the sorted list of nodes and their out degrees
    degreelist = G.out_degree()
    degreelist = list(degreelist)
    sorted_d = sorted((value, key) for [key,value] in degreelist)
    sources_degrees = [(value,key) for (value,key) in sorted_d if value >= 1]
    sources = [(key) for (value,key) in sorted_d if value >= 1]

    # building the set of sources. This is 1.5 times the training set size,
    # as we need "dummy" sources for fake edge generation.
    testsources_list = np.random.choice(sources,int(n_samples+n_samples/2))

    # these will be the edges with class 1
    testsources_real = testsources_list[0:int(n_samples/2)]

    # these will be the sources of the fake edges
    testsources_fake_1 = testsources_list[int(n_samples/2):int(n_samples)]

    # These will be used to generate the targets of the fake edges
    testsources_fake_2 = testsources_list[int(n_samples):int(n_samples+n_samples/2)]

    # Just formatting the sources into a list from an array
    testsources_1 = np.ndarray.tolist(testsources_real)
    testsources_2 = np.ndarray.tolist(testsources_fake_1)
    testsources = testsources_1 + testsources_2

    # initialize real and fake targets.
    targets_real = [0]*int(n_samples/2)
    targets_fake = [0]*int(n_samples/2)

    # loop to generate sources
    for i in range(0,int(n_samples/2)):
        q = 0
        # real sources are chosen uniformly from the successor sets
        targets_real[i] = np.random.choice(list(Gamma_out(testsources_real[i])))
        while q == 0:
            targets_fake[i] = np.random.choice(list(Gamma_out(testsources_fake_2[i])))

            # Bill: I think there is a small issue somewhere in this code below.
            # basically, what if the successor set of the node I'm generating from
            # is a subset of the successor set of the one I'm attaching it to?
            # Then we get stuck in the loop forever!

            # my solution is to just break the loop regardless - it seems the
            # probability of the edge being real is really low so the split ends up
            # being 50.1% to 49.9%, which isn't a big deal, but maybe we can improve

            # My other workaround is maybe to just break the loop after it iterates
            # over 10 times.

            if G.has_edge(testsources_fake_1[i],targets_fake[i]):
                q = 1
            else: q = 1

    # concatenate targets into one list.
    targets = targets_real + targets_fake

    # This appends everything into a data frame
    targetarray = np.array(targets)
    train = pd.DataFrame(testsources, columns = ['source'])
    train['target'] = targetarray
    train['class'] = train[['source','target']].apply(lambda x: int(G.has_edge(x['source'],x['target'])),axis=1)
    train = train.drop_duplicates(subset = ['source','target'])

    print('Train data generated. There are ', train['class'].sum(), ' unique real edges, and ',
            str(len(train) - train['class'].sum()), 'unique imaginary ones')


    return train
