"""Testing script."""
from minerminor import mm_generator as mmg
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr

base = mmg.learning_base_T21T_generation(6)
mmu.store_base(base, "base/T21T")
x = mmu.load_base("base/T21T")
lapl_set = mmr.labels_set_to_vec_laplacian_set(x)
print(lapl_set[0][0])
