"""Simple classification with scikit_learn."""
import minerminor.mm_utils as mmu
import minerminor.mm_representation as mmr
import sklearn as sk
import sklearn.tree as sk_tree
import networkx as nx

base_path = "bases/base_planar/learning-base-planar_18_[0, 1]_1000"
rep_adja = lambda x: mmr.graph_to_vec_adjacency(x)
representation_array = [rep_adja]
test_size = 0.2
model = sk_tree.DecisionTreeClassifier
save_path = ""

# On charge la base en mémoire.
learning_base = mmu.load_base(base_path)
# On applique la chaîne de représentation à la base
learning_base = mmr.learning_base_to_rep(learning_base, representation_array)
# On extrait les datarows et leurs labels associés
data_set, label_set = mmu.create_sample_label_classification(learning_base)
# Création des jeux d'entrainements et de tests
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(data_set, label_set, test_size=test_size)
# Instanciation du modèle de classification
clf = model()
# Apprentissage du modèle
clf.fit(x_train, y_train)
# Sauvegarde le modèle si le path de destination est définie
if save_path:
    model.save(save_path)
# Test du modèle sur l'apprentissage
y_pred = clf.predict(x_test)
# Génération de la matrice de confusion
mat_conf = sk.metrics.confusion_matrix(y_test, y_pred)
# Génération du rapport de test
report = sk.metrics.classification_report(y_test, y_pred, target_names=['P', '!P'])

print("Base_path : {0}".format(base_path))
print("Representation : {0}".format(representation_array))
print("Model : {0}".format(model))
print("Matrice de confusion : \n{0}".format(mat_conf))
print("{0}".format(report))
