"""Tools Box MinorMiner."""
import networkx as nx
import json
import matplotlib.pyplot as plt
import os
from networkx.readwrite import json_graph


def info_graph(graph):
    """Describe the argument graph."""
    print(nx.info(graph))
    print(nx.cycle_basis(graph))
    print(nx.laplacian_spectrum(graph))


def show_graph(graph):
    """Show the graph."""
    nx.draw(graph)
    plt.show()


def store_base(labels_set, label_set_name):
    """Store the learning base on JSON files."""
    if not os.path.exists(label_set_name):
        os.makedirs(label_set_name)
    for i, g in enumerate(labels_set):
        print("Ecriture des {0} graphes du label {1}.".format(len(g), i))
        file_path = "{0}/{1}_label_base.json".format(label_set_name, i)
        with open(file_path, "w") as f:
            json.dump(g, f)


def load_base(label_set_name):
    """Load the learning base from JSON files."""
    list_labels_files = os.listdir(label_set_name)
    labels_set = []
    for file_name in list_labels_files:
        path = "{0}/{1}".format(label_set_name, file_name)
        with open(path) as f:
            list_gaph = json.load(f)
            label_set = []
            for graph in list_gaph:
                label_set.append(json_graph.node_link_graph(graph))
            labels_set.append(label_set)

    return labels_set
