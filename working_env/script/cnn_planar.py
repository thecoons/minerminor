"""CNN for planar property recognition."""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from keras.utils import plot_model
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from minerminor import mm_draw as mmd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
import networkx as nx
import numpy as np
import keras


def model_init():
    """Init a Keras modele to be warp to scilearn."""
    # model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


# hyperparameters
batch_size = 128
num_classes = 2
epochs = 15
title = "model_test"
# input graph dimensions
graph_dim = 15

# load data
learning_base = mmu.load_base("base_planar_k5/learning-planar-minor_15_[0,1]_1000")

# representation
for count_classe, classe in enumerate(learning_base):
    for count_graph, graph in enumerate(classe):
        rep = nx.laplacian_matrix(graph).toarray()
        learning_base[count_classe][count_graph] = rep

# datarows formating
# -- extract from learning base et format it
data_set, label_set = mmu.create_sample_label_classification(learning_base)
x_train, x_test, y_train, y_test = train_test_split(data_set, label_set, test_size=0.2)
# -- list to np.array
x_train, x_test, y_train, y_test = [np.array(i) for i in [x_train, x_test, y_train, y_test]]
# -- reshape as tensorflow 'channels_last' input
x_train = x_train.reshape(x_train.shape[0], graph_dim, graph_dim, 1)
x_test = x_test.reshape(x_test.shape[0], graph_dim, graph_dim, 1)
# -- declaration input_shape (lis en global par la fonction de creation model... pas beau !)
input_shape = (graph_dim, graph_dim, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model init
model = KerasClassifier(build_fn=model_init, epochs=epochs, batch_size=batch_size, verbose=0)

# model fitting
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
mmd.plot_learning_curve(model, title, x_train, y_train, ylim=(0., 1.01), cv=cv, n_jobs=4)
# model.fit(x_train, y_train,
#           validation_data=(x_test, y_test))

# Visualisation
# plot_model(model, to_file='resultats/curve_pool/model_schema.png')

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
