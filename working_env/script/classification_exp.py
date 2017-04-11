"""Test tree classification."""
import networkx as nx
import numpy as np
from minerminor import mm_generator as mmg
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from sklearn import ensemble, svm, neighbors, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import os
import argparse
import datetime as dt

parser = argparse.ArgumentParser(prog="Script de classification")
parser.add_argument("-r", "--representation", nargs='*',
                    default=[mmr.adjacency,
                             mmr.laplacian,
                             mmr.A3_minus_D])
parser.add_argument("-c", "--classifieur", nargs='*',
                    default=[tree.DecisionTreeClassifier,
                             svm.SVC])
parser.add_argument("-p", "--path", default="base_from_jaguar",
                    help="Path de la learning base")

args = parser.parse_args()

resultat_file = open("resultats/{0}_{1}.txt".format(args.path, dt.datetime.now().time()), "a")

for directory in os.listdir(args.path):
    for classifieur in args.classifieur:
        for representation in args.representation:
            learning_base = mmu.load_base("{0}/{1}".format(args.path, directory))
            learning_base = representation(learning_base)
            # learning_base = mmr.labels_set_to_vec_laplacian_set(learning_base)
            # learning_base = mmr.labels_set_to_vec_adjacency_set(learning_base)
            # learning_base = mmr.learning_base_to_A3_minus_D(learning_base)

            data_set, label_set = mmu.create_sample_label_classification(learning_base)

            X_train, X_test, y_train, y_test = train_test_split(data_set, label_set,
                                                                test_size=0.1)
            # clf = svm.SVC()
            # clf = ensemble.RandomForestClassifier()
            # clf = neighbors.KNeighborsRegressor()
            clf = classifieur()

            pred, miss = mmu.learning(clf, X_train, X_test, y_train, y_test)

            accu, recal, fmeasure, _ = precision_recall_fscore_support(y_test,
                                                                       pred,
                                                                       average="macro")
            to_save = "|{0}|{1}|{2}|{3}%|{4}|{5}|{6}|\n".format(directory.replace("_", "|"),
                                                            classifieur.__name__,
                                                            representation.__name__,
                                                            miss,
                                                            accu,
                                                            recal,
                                                            fmeasure)
            print(directory, representation.__name__)
            print(to_save)
            resultat_file.write(to_save)
