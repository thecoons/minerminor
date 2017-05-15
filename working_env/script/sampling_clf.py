import networkx as nx
import minerminor.mm_utils as mmu
import minerminor.mm_draw as mmd
import minerminor.mm_generator as mmg

# G = nx.gnm_random_graph(20, 29)
# mmd.show_graph(G)
#
# for i in range(5):
#     G_ = mmu.graph_sampling(G, 10)
#
#     mmd.show_graph(G_)

lb = mmg.learning_base_rdm(10, None, 3)

mmd.show_graph(lb[1][0])
