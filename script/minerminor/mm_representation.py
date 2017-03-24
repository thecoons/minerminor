"""Representation function for MinorMiner."""
import numpy as np
import networkx as nx


def graph_to_vec_adjacency(graph):
    """Convert a graph to a vector from adjacency matrix."""
    mat = nx.to_numpy_matrix(graph)
    return np.squeeze(np.asarray(mat.reshape(-1)))


def graph_to_vec_laplacian(graph):
    """Convert a graph to a vector from laplacian matrix."""
    mat = nx.laplacian_matrix(graph).toarray()
    return np.squeeze(np.asarray(mat.reshape(-1)))


def graph_set_to_vec_adjacency_set(graph_set):
    """Convert a graph set to a vector adjacency set."""
    vec_adjacency_set = []
    for graph in graph_set:
        vec_adjacency_set.append(graph_to_vec_adjacency(graph))

    return vec_adjacency_set


def graph_set_to_vec_laplacian_set(graph_set):
    """Convert a graph set to a vector adjacency set."""
    vec_laplacian_set = []
    for graph in graph_set:
        vec_laplacian_set.append(graph_to_vec_laplacian(graph))

    return vec_laplacian_set


def labels_set_to_vec_adjacency_set(labels_set):
    """Convert labels set (set of graph set) to vector adjacency."""
    labels_vec_adjacency = []
    for graph_set in labels_set:
        labels_vec_adjacency.append(graph_set_to_vec_adjacency_set(graph_set))

    return labels_vec_adjacency


def labels_set_to_vec_laplacian_set(labels_set):
    """Convert labels set (set of graph set) to vector laplacian."""
    labels_vec_laplacian = []
    for graph_set in labels_set:
        labels_vec_laplacian.append(graph_set_to_vec_laplacian_set(graph_set))

    return labels_vec_laplacian
