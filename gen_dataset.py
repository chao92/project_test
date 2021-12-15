import collections
import random

from dataset_preprocess import gengraph
import matplotlib.pyplot as plt
import pickle
# G, labels, name = gengraph.gen_syn1()
import os
import networkx as nx
"""
BA-base (8 nodes)
House(5), Grid(4=2*2 nodes), Diamond(6 nodes), 
"""

def save_to_file(var_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)

def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list

def generate_syn_graph(base_num, shape_num, shape_type, added_edges, m, seed):

    G, labels, name = gengraph.gen_syn_shape(nb_shapes=shape_num, width_basis=base_num, \
                                        feature_generator=None, shape_type=shape_type,\
                                        add_random_edges=added_edges,m=m, seed=seed)

    print(" Graph is ", G, "type of G is", type(G))
    # print(" Lables is ", labels)
    # print(" edges is", list(G.edges), " with size = ", len(list(G.edges)))
    print(" name is ", name)
    name = name+"_seed_"+str(seed)
    nx.draw(G, with_labels=True, font_weight='bold')
    save_to_file(G, name+".pkl")
    plt.savefig(shape_type+"_m_"+str(m) + "_edge_"+str(len(list(G.edges)))+"_seed_"+str(seed)+".png")




def generate_super_graph(name):
    graph = gengraph.gen_super_graph(0, 30, m=2)
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.savefig("BA_shape.png")
    save_to_file(graph, name+".pkl")


def filter_super_graph(G, rules):
    print(" edges size = ", len(list(G.edges)))
    degrees = [(id, val) for (id, val) in G.degree()]
    print(" degree is ", type(degrees), degrees)
    degrees.sort(key=lambda x:x[1], reverse=True)
    print("sorted id by degrees: ", degrees)

    # for
    print(" G.nodes is", len(G.nodes()))
    size_nodes = len(G.nodes())
    mapping = [0] * size_nodes


    for i in range(len(degrees)):
        node_id, val = degrees[i]
        if i <= 4:
            mapping[node_id] = 2
        elif 5 <= i <= 9:
            mapping[node_id] = 5
        elif 10 <= i <= 19:
            if i % 2 == 0:
                mapping[node_id] = 3
            else:
                mapping[node_id] = 4
        elif 20 <= i <= 29:
            if i % 2 == 0:
                mapping[node_id] = 0
            else:
                mapping[node_id] = 1
    print(" labeling nodes with different types", mapping)
    print(" check edges")
    good_num = 0
    for edges in G.edges():
        print(edges)
        super_id_1, super_id_2 = mapping[edges[0]], mapping[edges[1]]
        if super_id_2 in rules[super_id_1] or super_id_1 in rules[super_id_2]:
            good_num += 1
    print(" good edges is", good_num)




def rules():

    dic = {
        0: [2],
        1: [2],
        2: [0, 1, 3, 4, 5],
        3: [2, 5],
        4: [2, 5],
        5: [2, 3, 4],
    }
    return dic

def generate_syn_dataset():
    # Barabási–Albert network must have m >= 1 and m < n, m = 2, n = 7
    base_num = 5
    shape_num = 1
    added_edges = 0
    shape_type = "house"
    # generating different shape
    for seed in range(5):
        generate_syn_graph(base_num, shape_num, shape_type="house", added_edges=0, m=1, seed=seed)
        generate_syn_graph(base_num, shape_num, shape_type="diamond",  added_edges=0, m=1, seed=seed)
        generate_syn_graph(base_num, shape_num, shape_type="cycle",  added_edges=0, m=1, seed=seed)
        generate_syn_graph(base_num, shape_num, shape_type="clique",  added_edges=0, m=1, seed=seed)
        generate_syn_graph(base_num, shape_num, shape_type="fan",  added_edges=0, m=1, seed=seed)
        generate_syn_graph(base_num, shape_num, shape_type="star",  added_edges=0, m=1, seed=seed)

if __name__ == '__main__':
    # name = "superGraph"
    # file_path = name + ".pkl"
    # if os.path.isfile(file_path):
    #     print(" load from file about generated super graph ")
    #     G = load_variable(file_path)
    # else:
    #     generate_super_graph(name)
    #
    # rules = rules()
    # filter_super_graph(G, rules)
    generate_syn_dataset()


