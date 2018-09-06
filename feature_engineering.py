import numpy as np
import pandas as pd
import networkx as nx
import random as random
import math as math
import time as time
import feature_list
import builtins

def generate_features(data):

    print('Begining feature computation.')

    t = time.time()
    data['common_friends'] = data[['source','target']].apply(lambda x: feature_list.common_friends(x['source'],x['target']),axis=1)
    print('common_friends feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['total_friends'] = data[['source','target']].apply(lambda x: feature_list.total_friends(x['source'],x['target']),axis=1)
    print('total_friends feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['pref_attach'] = data[['source','target']].apply(lambda x: feature_list.pref_attach(x['source'],x['target']),axis=1)
    print('pref_attach feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['jacard_coef'] = data[['source','target']].apply(lambda x: feature_list.jacard_coef(x['source'],x['target']),axis=1)
    print('jacard_coef feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['transitive_friends'] = data[['source','target']].apply(lambda x: feature_list.transitive_friends(x['source'],x['target']),axis=1)
    print('transitive_friends feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['opposite_friends'] = data[['source','target']].apply(lambda x: feature_list.opposite_friends(x['source'],x['target']),axis=1)
    print('opposite_friends feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['adar'] = data[['source','target']].apply(lambda x: feature_list.adar(x['source'],x['target']),axis=1)
    print('adar feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['dice_coef_directed'] = data[['source','target']].apply(lambda x: feature_list.dice_coef_directed(x['source'],x['target']),axis=1)
    print('dice_coef_directed feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['dice_coef_undirected'] = data[['source','target']].apply(lambda x: feature_list.dice_coef_undirected(x['source'],x['target']),axis=1)
    print('dice_coef_undirected feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['friends_closeness'] = data[['source','target']].apply(lambda x: feature_list.friends_closeness(x['source'],x['target']),axis=1)
    print('friends_closeness feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['jacard_outneighbours'] = data[['source','target']].apply(lambda x: feature_list.jacard_outneighbours(x['source'],x['target']),axis=1)
    print('jacard_outneighbours feature computed. Time taken: ',time.time() - t)

    t = time.time()
    data['jacard_inneighbours'] = data[['source','target']].apply(lambda x: feature_list.jacard_inneighbours(x['source'],x['target']),axis=1)
    print('jacard_inneighbours feature computed. Time taken: ',time.time() - t)
    print('Edge features computed. feature computed.')


    t = time.time()
    data['degree_source'] = data["source"].apply(lambda x: builtins.G.degree(x))
    data['degree_source_in'] = data["source"].apply(lambda x: builtins.G.in_degree(x))
    data['degree_source_out'] = data["source"].apply(lambda x: builtins.G.out_degree(x))
    data['degree_target'] = data["target"].apply(lambda x: builtins.G.degree(x))
    data['degree_target_in'] = data["target"].apply(lambda x: builtins.G.in_degree(x))
    data['degree_target_out'] = data["target"].apply(lambda x: builtins.G.out_degree(x))
    print('Node features computed. feature computed. Time taken: ',time.time() - t)

    print('All features computed.')
    return data
