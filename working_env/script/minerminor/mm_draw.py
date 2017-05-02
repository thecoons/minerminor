"""Librairie de graphique pour les minerminor"""
import csv
import collections as col
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np


def csv_to_dic(path_csv):
        cr = csv.reader(open(path_csv, 'r'), delimiter="|")
        res = {}
        # Gérer les méthodes de générations
        for row in cr:
            if row[5] not in res:
                res[row[5]] = {}
            if row[6] not in res[row[5]]:
                res[row[5]][row[6]] = {}
            res[row[5]][row[6]][row[4]] = row[10]

        return res


def create_fmeasure_curve(dict_methodes):
    cycol = cycle(['r--', 'b-.', 'g:'])
    count = 1
    plt.subplots_adjust(hspace=0.9)
    plt.figure(1)
    for key, value in dict_methodes.items():
        plt.subplot(len(dict_methodes), 1, count)
        arr_legend = []
        for key_, value_ in value.items():
            od = {int(k): v for k, v in value_.items()}
            t1 = [i for i, v in sorted(od.items())]
            t2 = [v for i, v in sorted(od.items())]
            f_, = plt.plot(t1, t2, next(cycol), label=key_)
            arr_legend.append(f_)
            plt.axis([None, None, 0., 1.])
            plt.xlabel('Features Size')
            plt.ylabel('F-Measure')
            plt.title(key)
        count += 1
    plt.legend(loc=1, borderaxespad=-10., handles=arr_legend, fontsize='small')
    # plt.legend(bbox_to_anchor=(1, 1), loc=1,
    #            ncol=2, mode="expand", borderaxespad=0., handles=arr_legend)
    # plt.legend(handles=arr_legend, loc='lower')
    plt.show()


def create_spread_curve(data):
    """Create the curve for a data base gen."""
    cycol = cycle(['r--', 'b-.', 'g:'])
    arr_legend = []
    for count_classe, key_classe in enumerate(data.keys()):
        plt.figure(count_classe+1)
        for count_eigen, keys_eigen in enumerate(data[key_classe].keys()):
            # print(eigen_keys)
            plt.subplot(len(data[key_classe].keys()), 1, count_eigen + 1)
            # print(len(data[key_classe].keys()), 1, count_classe + 1)
            for method in data[key_classe][keys_eigen].keys():
                # print(method)
                t1 = [i for i, v in sorted(data[key_classe][keys_eigen][method].items())]
                t2 = [v for i, v in sorted(data[key_classe][keys_eigen][method].items())]
                f_, = plt.plot(t1, t2, next(cycol), label=method)
                arr_legend.append(f_)
            plt.xlabel('Features Size')
            plt.ylabel('Mean')
    plt.legend(loc=1, borderaxespad=-10., handles=arr_legend, fontsize='small')
    plt.show()


def create_curve_xP(data):
    cycol = cycle(['r--', 'b-.', 'g:'])
    arr_legend = []
    plt.figure(1)
    for count_meth, meth in enumerate(data.keys()):
        print(meth)
        t1 = [i for i, v in sorted(data[meth].items())]
        t2 = [v for i, v in sorted(data[meth].items())]
        print(t1)
        print(t2)
        f_, = plt.plot(t1, t2, next(cycol), label=meth)
        arr_legend.append(f_)
    plt.xlabel('0-P_vs_n-P')
    plt.ylabel('Miss')
    plt.legend(loc=1, borderaxespad=-10., handles=arr_legend, fontsize='small')
    plt.show()
