"""CNN for planar property recognition."""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from minerminor import mm_utils as mmu
from minerminor import mm_representation as mmr
from sklearn.model_selection import train_test_split
import networkx as nx
import numpy as np
import keras

# hyperparameters
batch_size = 128
num_classes = 2
epochs = 12

# input graph dimensions
graph_dim = 15

# load data
learning_base = mmu.load_base("base_planar_k5k33/learning-planar-minor_15_[0,1]_10000")

# representation
for count_classe, classe in enumerate(learning_base):
    for count_graph, graph in enumerate(classe):
        learning_base[count_classe][count_graph] = nx.laplacian_matrix(graph).toarray()

# datarows formating
# -- extract from learning base et format it
data_set, label_set = mmu.create_sample_label_classification(learning_base)
x_train, x_test, y_train, y_test = train_test_split(data_set, label_set, test_size=0.2)
# -- list to np.array
x_train, x_test, y_train, y_test = [np.array(i) for i in [x_train, x_test, y_train, y_test]]
# -- reshape as tensorflow 'channels_last' input
x_train = x_train.reshape(x_train.shape[0], graph_dim, graph_dim, 1)
x_test = x_test.reshape(x_test.shape[0], graph_dim, graph_dim, 1)
# -- declaration input_shape
input_shape = (graph_dim, graph_dim, 1)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
