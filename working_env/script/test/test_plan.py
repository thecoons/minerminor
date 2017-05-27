"""Script test."""
import minerminor.mm_draw as mmd
import minerminor.mm_generator as mmg
import planarity as pl

# G, H = mmg.generate_planar_deg(8, 0)
#
# print(pl.is_planar(G), pl.is_planar(H))
# mmd.show_graph(G)
# mmd.show_graph(H)

learning_base = mmg.learning_base_planar(10, [1, 2], 4)

for count, i in enumerate(learning_base[0]):
    mmd.show_graph(learning_base[0][count])
    mmd.show_graph(learning_base[1][count])
