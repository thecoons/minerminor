"""Script split TW."""
import minerminor.mm_utils as mmu
import os

learning_base = []
path_base = "base_tw_2"
list_labels_dir = os.listdir(path_base)
for dir_name in list_labels_dir:
    base = mmu.tw_split_base(path_base+"/"+dir_name)
    mmu.store_base(base, path_base+"_transf/"+dir_name)

# for base in learning_base:
# print(learning_base)
