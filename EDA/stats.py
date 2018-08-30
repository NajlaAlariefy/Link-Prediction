import networkx as nx

def get_network_statistics(g):
    num_nodes = nx.number_of_nodes(g)
    num_edges = nx.number_of_edges(g)
    density = nx.density(g)
    transitivity = nx.transitivity(g)


    network_statistics = {
        'num_nodes':num_nodes,
        'num_edges':num_edges,
        'density':density,
        'transitivity':transitivity
    }

    return network_statistics

def save_network_statistics(g, file_name):
    network_statistics = get_network_statistics(g)
    with open(file_name, 'wb') as f:
        pickle.dump(network_statistics, f)
#save_network_statistics(G, 'network_statistics.txt')
