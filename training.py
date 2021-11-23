import numpy as np
import cv2
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.models import load_model
import pickle

from sklearn.preprocessing import MinMaxScaler

np.seterr(divide='ignore', invalid='ignore')


def run_program():
    array_x, array_Y = import_data()
    classified = run_network(array_x, array_Y)
    graph_stats(classified)


def import_data():
    # Loading Data
    x_array_in = open("sample_3.array", "rb")
    array_X = pickle.load(x_array_in)

    y_array_in = open("label_3.array", "rb")
    Y = pickle.load(y_array_in)

    # array_X = np.array(X)
    # array_X = array_X.squeeze()
    array_Y = np.asarray(Y, dtype='int16').reshape((-1, 1))
    array_Y = tf.keras.utils.to_categorical(array_Y, 8)

    # Create Scalar
    scaler = MinMaxScaler()

    for i in range(len(array_X)):
        # Reduce dimensions
        array_X[i] = array_X[i][:, :, 1]
        # Normalize values to be between 0 and 1
        array_X[i] = scaler.fit_transform(array_X[i])
        # Re-merge image so that it is 3 dimensional
        array_X[i] = cv2.merge((array_X[i], array_X[i], array_X[i]), -1)

    # Convert to Numpy Array for input to network
    array_X = np.array(array_X)


    index = 0
    index1 = 0

    # Counting the number of images within each category
    # for x in Y:
    #     if x == 1:
    #         index += 1
    #     elif x == 0:
    #         index1 += 1
    # print(index)
    # print(index1)

    print(len(array_Y))

    return array_X, array_Y


def run_network(array_X, array_Y):
    # classifier = Sequential()
    # classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=array_X.shape[1:]))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=array_X.shape[1:]))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=array_X.shape[1:]))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=array_X.shape[1:]))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # classifier.add(Flatten())
    #
    # classifier.add(Dense(units=512, activation='relu'))
    # classifier.add(Dense(units=8, activation='softmax'))
    # # #
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model = Sequential()
    # model.add(BatchNormalization())
    model.add(Conv2D(input_shape=array_X.shape[1:], filters=64, kernel_size=(3, 3), padding="same", activation="relu"
                     , kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=l2(0.0005),
                     bias_regularizer=l2(0.0005)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=l2(0.0005)
    #                  , bias_regularizer=l2(0.0005)))
    # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(BatchNormalization())
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    # model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dense(units=8, activation="softmax"))
    # # #
    mcp_save = ModelCheckpoint('cell_cnn_60x', save_best_only=True, monitor='val_acc', mode='min')
    #
    # classifier.compile(loss='categorical_crossentropy',
    #                    optimizer=opt,
    #                    metrics=['acc'])
    #
    # classifier_history = classifier.fit(array_X, array_Y, batch_size=32, epochs=20, validation_split=0.3, callbacks=[mcp_save])
    # print(classifier)
    # print(classifier.summary())
    # classifier.save('cell_cnn_60xsample.h5')

    # for i in range(5):
    #     time.sleep(30)
    #
    #     del classifier
    #
    #     classifier = load_model('cell_cnn_60xsample.h5')
    #     classified = classifier.fit(array_X, array_Y, batch_size=32, epochs=3, validation_split=0.3)
    #     classifier.save('cell_cnn_60xsample.h5')
    #     if i == 4:
    #         time.sleep(30)
    #         classifier.save('cell_cnn_60xsample.h5')
    #
    #         del classifier
    #
    #         classifier = load_model('cell_cnn_60xsample.h5')
    #         classified = classifier.fit(array_X, array_Y, batch_size=32, epochs=3, validation_split=0.3)
    #         return classifier, classified

    # return classifier_history

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    # ModelCheckpoint(filepath=,
    #                                monitor = 'val_acc',
    #                                verbose=
    #                                save_best_only=True)

    classified = model.fit(array_X, array_Y, batch_size=32, epochs=1, validation_split=0.3, callbacks=[mcp_save])
    model.save('cell_cnn_60xsample.h5')
    print(model.summary())

    models = load_model('cell_cnn_60xsample.h5')
    x = array_X[:100]
    y = array_Y[:100]

    res = models.predict(x)

    tf.math.confusion_matrix(y, res)


    # # for index in range(5):
    # #     time.sleep(450)
    # #     del model
    # #
    # #     model = load_model('cell_cnn_60xsample.h5')
    # #     classified = model.fit(array_X, array_Y, batch_size=32, epochs=3, validation_split=0.3, callbacks=[mcp_save])
    # #     model.save('cell_cnn60xsample.h5')
    # #     if index == 4:
    # #         time.sleep(450)
    # #         del model
    # #
    # #         model = load_model('cell_cnn_60xsample.h5')
    # #         classified = model.fit(array_X, array_Y, batch_size=32, epochs=3, validation_split=0.3,
    # #                                callbacks=[mcp_save])
    # #         return model, classified


    return classified


def graph_stats(classified):
    acc = classified.history['acc']
    val_acc = classified.history['val_acc']
    # loss = classified.history['loss']
    # val_loss = classified.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    plt.savefig('model_stats.png')


run_program()
