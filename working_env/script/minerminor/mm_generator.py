"""Training Base Generator."""

import networkx as nx
from networkx.readwrite import json_graph as js
import random as rdm
from progress_bar import InitBar as ib
import math
from minerminor import mm_utils as mmu


def choice_first_node(graph):
    """Pick up a node which can form a cycle and a list for the seconde one."""
    node = rdm.choice(graph.nodes())
    possible_nodes_iter = nx.non_neighbors(graph, node)
    possible_nodes = []
    for i in possible_nodes_iter:
        possible_nodes.append(i)

    if possible_nodes:
        return node, possible_nodes
    else:
        return choice_first_node(graph)


def tree_to_1tree(graph):
    # A update avec la fonction nx.non_edge() ...
    """Transform a tree to 1-tree."""
    first_node, possible_nodes = choice_first_node(graph)
    seconde_node = rdm.choice(possible_nodes)
    graph.add_edge(first_node, seconde_node)

    return graph


def learning_base_T21T_generation(rank):
    """Generate T21T learning base with nonisomorphic tree for a given rank."""
    data_set = nx.nonisomorphic_trees(rank, create="graph")
    tree_set, cycle_set = [], []
    pbar = ib()
    total = nx.number_of_nonisomorphic_trees(rank)
    for count, i in enumerate(data_set):
        pbar(count/total * 100)
        tree_set.append(js.node_link_data(i))
        cycle_set.append(js.node_link_data(tree_to_1tree(i)))

    return [tree_set, cycle_set]


def pTree_generation(tree, arr_ptree_rank):
    """From tree make P_tree."""
    # On vérifie le nombre d'arrêtes disponble pour borner max_cycle.
    edges_set = []
    max_ptree_rank = max(arr_ptree_rank) + 1
    for edge in nx.non_edges(tree):
        edges_set.append(edge)
    if len(edges_set) < max_ptree_rank:
        max_ptree_rank = len(edges_set)
    # On construit en ensemble de pTree avec un arbre en seed.
    ptree_set = []
    current_graph = tree
    for i in range(max_ptree_rank):
        if i in arr_ptree_rank:
            ptree_set.append(nx.Graph(current_graph))
        current_graph = tree_to_1tree(current_graph)

    return ptree_set


def pTree_basic_cycle_generation(nb_nodes, arr_ptree_rank, depth_base):
    """P-Tree Learning base generator with Euler lem."""
    learning_base = []
    for p_rank in arr_ptree_rank:
        ptree_class = []
        # nb_edges = nb_nodes + p_rank - 1
        pbar = ib()
        print("Construction de la classe {0}-Tree : {1} à construire".format(
            p_rank, depth_base))
        tree_set = nx.nonisomorphic_trees(nb_nodes)
        while len(ptree_class) < depth_base:
            try:
                graph = next(tree_set)
            except StopIteration:
                tree_set = nx.nonisomorphic_trees(nb_nodes)
                graph = next(tree_set)

            for i in range(p_rank):
                non_edges = [[x, y] for x, y in nx.non_edges(graph)]
                x, y = rdm.choice(non_edges)
                graph.add_edge(x, y)

            if not mmu.robust_iso(graph, ptree_class):
                pbar(len(ptree_class)/depth_base*100)
                ptree_class.append(graph)

        learning_base.append(ptree_class)

    return learning_base


def learning_base_pTree_generation(nb_nodes, arr_ptree_rank, feature_size):
    """Generate PTree classes with nonisomorphic tree for a given rank."""
    limit_edges = int((math.pow(nb_nodes, 2)-3*nb_nodes+2)/2)
    max_ptree_rank = max(arr_ptree_rank)
    if limit_edges < max_ptree_rank:
        print("Attention la limite de cycle basic est de {}".format(
                limit_edges))
        max_ptree_rank = limit_edges
    learning_base = [[] for i in arr_ptree_rank]
    tree_itr = nx.nonisomorphic_trees(nb_nodes, create="graph")
    pbar = ib()
    print("""Construction des {0} features en parallèles.
          """.format(len(arr_ptree_rank)*feature_size))
    for count, tree in enumerate(tree_itr):
        pbar(count/feature_size*100)
        for i, graph in enumerate(pTree_generation(tree, arr_ptree_rank)):
            learning_base[i].append(graph)
        if count >= feature_size:
            break

    return learning_base


def learning_base_tw2():
    """Generate TW2 learning base."""
    pass
