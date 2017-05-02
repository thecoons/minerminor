from sklearn.externals import joblib
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from minerminor import mm_draw as mmd
from sklearn.model_selection import train_test_split
import os
from os.path import isfile, join
mypath = "classifier/class_test"
res = {}
for clf_file in [f for f in os.listdir(mypath) if isfile(join(mypath, f))]:
    clf = joblib.load(mypath+"/"+clf_file)
    arr_info_clf = clf_file.split("_")
    res[arr_info_clf[0]] = {}
    for base_dir in os.listdir("base_spread"):
        learning_base = mmu.load_base("base_spread/"+base_dir)
        arr_info_base = base_dir.split("_")
        arr_class = arr_info_base[2][1:-1].split(",")
        # mmu.show_graph(learning_base[1][0])
        learning_base = mmr.adjacency(learning_base)

        data_set, label_set = mmu.create_sample_label_classification(learning_base)
        for count, label in enumerate(label_set):
            if label > 0:
                label_set[count] = 1
        # print(label_set)
        # X_train, X_test, y_train, y_test = train_test_split(data_set, label_set,
        #                                                     test_size=0.1)
        y_pred = clf.predict(data_set)
        miss = (label_set != y_pred).sum()
        total = len(y_pred)

        # print("Number of mislabeled on a total %d : %d (%d %%)" % (
        #     total, miss, (miss/total)*100))
        res[arr_info_clf[0]][int(arr_class[1])] = miss
print(res)
mmd.create_curve_xP(res)
# générer la data
# draw la data
