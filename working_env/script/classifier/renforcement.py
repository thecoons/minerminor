from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from minerminor import mm_representation as mmr
from minerminor import mm_utils as mmu


clf_path = "classifier/clf_base_validation_tw1/MLPClassifier_adjacency_2000.pkl"
base_renf_path = "bases/base_rdm_kp/learning-base-rdm_14_[0, 1]_1000"
representation = mmr.adjacency

clf = joblib.load(clf_path)

learning_base = mmu.load_base(base_renf_path)
learning_base = representation(learning_base)

data_set, label_set = mmu.create_sample_label_classification(learning_base)

y_pred = cross_val_predict(clf, data_set, label_set, cv=10)

mat_conf = confusion_matrix(label_set, y_pred)
report = classification_report(label_set, y_pred, target_names=['P', '!P'])

print("{0}|{1}|\n\n{2}|\n{3}|\n{4}".format(base_renf_path,
                                           representation.__name__,
                                           clf,
                                           mat_conf.tolist(),
                                           report))
