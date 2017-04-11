"""Ptree Test script."""
import networkx as nx
import numpy as np
from minerminor import mm_generator as mmg
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from sklearn import ensemble, svm, neighbors, tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pydotplus
import time
from sklearn.metrics import precision_recall_fscore_support

# mmu.experiment_generation([mmg.pTree_basic_cycle_generation,
#                           mmg.learning_base_pTree_generation],
#                           [100],
#                           [5, 20, 50],
#                           [100])


# learning_base = mmg.learning_base_pTree_generation(11, 3)
# print(len(learning_base))
# mmu.show_graph(learning_base[0][0])
# mmu.show_graph(learning_base[1][0])
# learning_base = mmg.pTree_basic_cycle_generation(8, np.arange(5, 7), 1000)
# mmu.store_base(learning_base, "base/ptree_8_5-7_1000/")
#
# for g_set in learning_base:
#     mmu.show_graph(g_set[0])
path = "base_12/pTree_basic_cycle_generation_12_[0, 1]_500"
learning_base = mmu.load_base(path)

# mmu.show_graph(learning_base[0][0])

# learning_base = mmr.labels_set_to_vec_laplacian_set(learning_base)
# learning_base = mmr.labels_set_to_vec_adjacency_set(learning_base)
learning_base = mmr.learning_base_to_A3_minus_D(learning_base)
#
data_set, label_set = mmu.create_sample_label_classification(learning_base)
#
X_train, X_test, y_train, y_test = train_test_split(data_set, label_set,
                                                    test_size=0.1)
# clf = svm.SVC()
# clf = ensemble.RandomForestClassifier()
# clf = neighbors.KNeighborsRegressor()
clf = tree.DecisionTreeClassifier()

pred, _ = mmu.learning(clf, X_train, X_test, y_train, y_test)
print(len(y_test), len(pred))
print(precision_recall_fscore_support(y_test, pred, average="macro"))
dot_data = tree.export_graphviz(clf, out_file=None,
                                class_names=True,
                                filled=True, rounded=True,
                                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("tree.pdf")
print(X_test, y_test)

# plt.subplot(2, 1, 0 + 1)
# plt.scatter(X_train, y_train, c='k', label='data')
# plt.plot(X_test, y_test, c='g', label='prediction')
# plt.axis('tight')
# plt.legend()
#
# plt.show()
