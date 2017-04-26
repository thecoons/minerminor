"""P_tree generation script and benchmarking."""
# import networkx as nx
from minerminor import mm_utils as mmu
from minerminor import mm_generator as mmg
import argparse

parser = argparse.ArgumentParser(prog="P-Tree Generation")
parser.add_argument("-g", "--arr_generators", nargs='*',
                    default=[mmg.pTree_basic_cycle_generation])
parser.add_argument("-n", "--arr_rank_nodes", default=[10], nargs='*',
                    help="Nodes array", type=int)
parser.add_argument("-s", "--arr_features_size", default=[10], nargs='*',
                    help="Features depth", type=int)
parser.add_argument("-r", "--arr_ptree_rank", default=[0, 1], nargs='*',
                    help="P-Trees rank", type=int)
parser.add_argument("-p", "--path_dir", default="base_test",
                    help="Path of base dir")

args = parser.parse_args()

learning_base = mmu.experiment_generation(args.arr_generators,
                                          args.arr_rank_nodes,
                                          args.arr_ptree_rank,
                                          args.arr_features_size,
                                          args.path_dir)
