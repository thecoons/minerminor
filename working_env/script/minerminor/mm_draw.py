"""Librairie de graphique pour les minerminor"""
import csv
import collections as col
import matplotlib.pyplot as plt
from itertools import cycle


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
