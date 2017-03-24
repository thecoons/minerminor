"""Testing script."""
from minerminor import mm_generator as mmg
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from sklearn.model_selection import train_test_split
from sklearn import ensemble


base = mmg.learning_base_T21T_generation(15)
mmu.store_base(base, "base/T21T")
x = mmu.load_base("base/T21T")
lapl_set = mmr.labels_set_to_vec_laplacian_set(x)

data_set, label_set = mmu.create_sample_label_classification(lapl_set)

X_train, X_test, y_train, y_test = train_test_split(data_set, label_set,
                                                    test_size=0.2)

clf = ensemble.RandomForestClassifier()
y_pred = clf.fit(X_train, y_train).predict(X_test)

miss = (y_test != y_pred).sum()

print("Number of mislabeled points out of a total %d points : %d (%d %%)" % (
    len(X_test), miss, (miss/len(X_test))*100))
