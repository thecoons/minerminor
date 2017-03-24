"""Training Base Generator."""

import networkx as nx
from networkx.readwrite import json_graph as js
import random as rdm
from progress_bar import InitBar as ib


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
