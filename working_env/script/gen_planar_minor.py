"""Script."""
import minerminor.mm_generator as mmg
import minerminor.mm_utils as mmu
import minerminor.mm_draw as mmd
import networkx as nx
import random as rdm

K5 = nx.complete_graph(5)
K33 = nx.complete_bipartite_graph(3, 3)
lb = mmg.learning_base_planar_by_minor_agreg(15, 1000, K33)
lb2 = mmg.learning_base_planar_by_minor_agreg(15, 1000, K5)

lb3 = [[], []]

lb3[0].extend(rdm.sample(lb[0], 500))
lb3[0].extend(rdm.sample(lb2[0], 500))

lb3[1].extend(rdm.sample(lb[1], 500))
lb3[1].extend(rdm.sample(lb2[1], 500))

mmu.store_base(lb3, "base_planar_k5k33")
mmd.show_graph(lb[0][1])
mmd.show_graph(lb[1][1])
