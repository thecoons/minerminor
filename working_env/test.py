"""Testing script."""
from minerminor import mm_generator as mmg
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
import networkx as nx
import numpy as np

graph = nx.from_numpy_matrix(np.matrix([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]]))
print(mmr.graph_to_A3_minus_D(graph))
