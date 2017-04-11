"""Tools Box MinorMiner."""
import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from networkx.readwrite import json_graph
import time
import csv
import datetime
from sklearn.model_selection import learning_curve


def count_iterable(i):
    """len() pour un iterable."""
    return sum(1 for e in i)


def info_graph(graph):
    """Describe the argument graph."""
    print(nx.info(graph))
    print(nx.cycle_basis(graph))
    print(nx.laplacian_spectrum(graph))


def show_graph(graph):
    """Show the graph."""
    pos = nx.nx_pydot.graphviz_layout(graph)
    nx.draw(graph, pos=pos)
    plt.show()


def fast_iso(graph, graph_set):
    """Fast iso test."""
    for g_test in graph_set:
        if nx.could_be_isomorphic(graph, g_test):
            return True
    return False


def robust_iso(graph, graph_set):
    """Fast iso test."""
    for g_test in graph_set:
        if nx.is_isomorphic(graph, g_test):
            return True
    return False


def graph_set_to_json(graph_set):
    """Convert graph set to json set."""
    json_set = []
    for g in graph_set:
        json_set.append(json_graph.node_link_data(g))

    return json_set


def json_to_graph_set(json_set):
    """Convert json set to graph set."""
    graph_set = []
    for j in json_set:
        graph_set.append(json_graph.node_link_graph(j))

    return graph_set


def store_base(labels_set, label_set_name):
    """Store the learning base on JSON files."""
    if not os.path.exists(label_set_name):
        os.makedirs(label_set_name)
    for i, g in enumerate(labels_set):
        print("Ecriture des {0} graphes du label {1}.".format(len(g), i))
        file_path = "{0}/{1}_label_base.json".format(label_set_name, i)
        with open(file_path, "w") as f:
            json.dump(graph_set_to_json(g), f)


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


def exp_to_csv(generator, nb_nodes, feature_size, ptree_rank, time_exe):
    """Write the csv of experiment."""
    c = csv.writer(open("{0}.csv".format(datetime.datetime.now)), "wb")
    c.writerow([generator, nb_nodes, feature_size, ptree_rank, time_exe])


def create_sample_label_classification(labels_set):
    """Create data and label set."""
    data_set, label_set = [], []
    for i, graph_set in enumerate(labels_set):
        for graph in graph_set:
            data_set.append(graph)
            label_set.append(i)

    return data_set, label_set


def learning(classifier, X_train, X_test, y_train, y_test):
    """Learning and Predict function."""
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    miss = (y_test != y_pred).sum()
    total = len(X_test)
    print("Number of mislabeled on a total %d : %d (%d %%)" % (
        total, miss, (miss/total)*100))

    return y_pred, (miss/total)*100


def experiment_generation(arr_generator, arr_nb_nodes, arr_ptree_rank,
                          arr_features_size, path_dir):
    """Experiment methode for base learning generation."""
    for generator in arr_generator:
        for nb_nodes in arr_nb_nodes:
            for feature_size in arr_features_size:
                t0 = time.time()
                learning_base = generator(nb_nodes,
                                          arr_ptree_rank,
                                          feature_size)
                t1 = time.time()
                path = "{0}/{1}_{2}_{3}_{4}/".format(path_dir,
                                                     generator.__name__.replace("_", "-"),
                                                     nb_nodes,
                                                     arr_ptree_rank,
                                                     feature_size)
                store_base(learning_base, path)
                t2 = time.time()
                print("""
~~~~~~~~~
Gen : {0},\nNb_nd : {1},\nP-T_rk : {2},\nF_size : {3}, #F : {4}\n
    => Time build : {5} sec, Time store : {6} sec
#########        """.format(generator.__name__,
                            nb_nodes,
                            arr_ptree_rank,
                            feature_size,
                            len(learning_base),
                            str(t1-t0),
                            str(t2-t1)))
                print("|{0}|{1}|{2}|{3}|{4}|".format(generator.__name__,
                                                     nb_nodes,
                                                     feature_size,
                                                     arr_ptree_rank,
                                                     str(t1-t0)))
                # for i, ptree in enumerate(arr_ptree_rank):
                #     show_graph(learning_base[i][0])



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
