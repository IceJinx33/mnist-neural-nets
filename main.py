#!/usr/bin/env python3
import os
import numpy
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

import tensorflow as tf
mnist = tf.keras.datasets.mnist

from tensorflow import keras
from tensorflow.keras import layers, activations
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from scipy.stats import loguniform, uniform

### hyperparameter settings and other constants
batch_size = 128
num_classes = 10
epochs = 10
mnist_input_shape = (28, 28, 1)
d1 = 1024
d2 = 256
alpha = 0.1
beta = 0.9
alpha_adam = 0.001
rho1 = 0.99
rho2 = 0.999
### end hyperparameter settings


# load the MNIST dataset using TensorFlow/Keras
def load_MNIST_dataset():
    mnist = tf.keras.datasets.mnist
    (Xs_tr, Ys_tr), (Xs_te, Ys_te) = mnist.load_data()
    Xs_tr = Xs_tr / 255.0
    Xs_te = Xs_te / 255.0
    Xs_tr = Xs_tr.reshape(Xs_tr.shape[0], 28, 28, 1) # 28 rows, 28 columns, 1 channel
    Xs_te = Xs_te.reshape(Xs_te.shape[0], 28, 28, 1)
    return (Xs_tr, Ys_tr, Xs_te, Ys_te)


# evaluate a trained model on MNIST data, and print the usual output from TF
#
# Xs        examples to evaluate on
# Ys        labels to evaluate on
# model     trained model
#
# returns   tuple of (loss, accuracy)
def evaluate_model(Xs, Ys, model):
    (loss, accuracy) = model.evaluate(Xs, Ys, verbose = 0)
    return (loss, accuracy)


# train a fully connected two-hidden-layer neural network on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# beta      momentum parameter (0.0 if no momentum)
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, B, epochs):

    model = keras.Sequential()
    model.add(keras.Input(shape = mnist_input_shape, name = "input"))
    model.add(layers.Flatten())
    model.add(layers.Dense(d1, activation = "relu", name = "dense_1"))
    model.add(layers.Dense(d2, activation = "relu", name = "dense_2"))
    model.add(layers.Dense(num_classes, activation = "softmax", name = "last"))
    
    model.compile(
    optimizer = keras.optimizers.SGD(learning_rate = alpha, momentum = beta),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    start = time.time()
    hist = model.fit(x = Xs, y = Ys, batch_size = B, epochs = epochs, validation_split = 0.1)
    end = time.time()

    hist.history['runtime'] = end - start
    return (model, hist)


# train a fully connected two-hidden-layer neural network on MNIST data using Adam, and print the usual output from TF
# 
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# rho1      first moment decay parameter
# rho2      second moment decay parameter
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_adam(Xs, Ys, d1, d2, alpha, rho1, rho2, B, epochs):

    model = keras.Sequential()
    model.add(keras.Input(shape = mnist_input_shape, name = "input"))
    model.add(layers.Flatten())
    model.add(layers.Dense(d1, activation = "relu", name = "dense_1"))
    model.add(layers.Dense(d2, activation = "relu", name = "dense_2"))
    model.add(layers.Dense(num_classes, activation = "softmax", name = "last"))
    
    model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = alpha, beta_1 = rho1, beta_2 = rho2),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    start = time.time()
    hist = model.fit(x = Xs, y = Ys, batch_size = B, epochs = epochs, validation_split = 0.1)
    end = time.time()

    hist.history['runtime'] = end - start
    return (model, hist)


# train a fully connected two-hidden-layer neural network with Batch Normalization on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# beta      momentum parameter (0.0 if no momentum)
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_bn_sgd(Xs, Ys, d1, d2, alpha, beta, B, epochs):
    
    model = keras.Sequential()
    model.add(keras.Input(shape = mnist_input_shape, name = "input"))
    model.add(layers.Flatten())
    model.add(layers.Dense(d1, name = "dense_1"))
    model.add(layers.BatchNormalization(name = "bn1"))
    model.add(layers.Activation(activations.relu, name = "relu1"))
    model.add(layers.Dense(d2, name = "dense_2"))
    model.add(layers.BatchNormalization(name = "bn2"))
    model.add(layers.Activation(activations.relu, name = "relu2"))
    model.add(layers.Dense(num_classes, name = "last"))
    model.add(layers.BatchNormalization(name = "bnlast"))
    model.add(layers.Activation(activations.softmax, name = "softmaxlast"))
    
    model.compile(
    optimizer = keras.optimizers.SGD(learning_rate = alpha, momentum = beta),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    start = time.time()
    hist = model.fit(x = Xs, y = Ys, batch_size = B, epochs = epochs, validation_split = 0.1)
    end = time.time()

    hist.history['runtime'] = end - start
    return (model, hist)


# train a convolutional neural network on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# alpha     step size parameter
# rho1      first moment decay parameter
# rho2      second moment decay parameter
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_CNN_sgd(Xs, Ys, alpha, rho1, rho2, B, epochs):
    
    model = keras.Sequential()
    model.add(keras.Input(shape = mnist_input_shape, name = "input"))
    model.add(layers.Conv2D(filters = 32, kernel_size = (5,5), activation = "relu", name = "conv1"))
    model.add(layers.MaxPool2D(pool_size = (2,2), name = "maxpool1"))
    model.add(layers.Conv2D(filters = 64, kernel_size = (5,5), activation = "relu", name = "conv2"))
    model.add(layers.MaxPool2D(pool_size = (2,2), name = "maxpool2"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation = "relu", name = "dense_1"))
    model.add(layers.Dense(num_classes, activation = "softmax", name = "last"))
    
    model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = alpha, beta_1 = rho1, beta_2 = rho2),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    start = time.time()
    hist = model.fit(x = Xs, y = Ys, batch_size = B, epochs = epochs, validation_split = 0.1)
    end = time.time()

    hist.history['runtime'] = end - start
    return (model, hist)


# plots the loss and accuracy graphs versus number of epochs of a model given its history
#
# histo          the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
# model_name     a string representing the name of the model
def plot_graphs(histo, model_name):
    pyplot.figure(figsize=(11,8))
    
    hist = histo.history
    # Loss vs Number of Epochs
    pyplot.plot(numpy.arange(1, 11), hist['loss'], "-r")
    pyplot.plot(numpy.arange(1, 11), hist['val_loss'], "-g")
    pyplot.plot(numpy.arange(1, 11), [hist['test_loss']]*10, "-b")
    pyplot.xticks(numpy.arange(1, 11), numpy.arange(1,11))
    pyplot.legend(['Training Loss', 'Validation Loss', 'Final Test Loss'])
    pyplot.xlabel('Number of Epochs', labelpad=15)
    pyplot.ylabel('Loss', labelpad=15)
    pyplot.title(model_name + " - Loss vs Number of Epochs", pad=15)
    pyplot.savefig(model_name.replace(" ", "") + "_loss.png", dpi=300)
    pyplot.clf()

    # Accuracy vs Number of Epochs
    pyplot.plot(numpy.arange(1, 11), hist['sparse_categorical_accuracy'], "-r")
    pyplot.plot(numpy.arange(1, 11), hist['val_sparse_categorical_accuracy'], "-g")
    pyplot.plot(numpy.arange(1, 11), [hist['test_accuracy']]*10, "-b")
    pyplot.xticks(numpy.arange(1, 11), numpy.arange(1,11))
    pyplot.legend(['Training Accuracy', 'Validation Accuracy', 'Final Test Accuracy'])
    pyplot.xlabel('Number of Epochs', labelpad=15)
    pyplot.ylabel('Accuracy (Fraction)', labelpad=15)
    pyplot.title(model_name + " - Accuracy vs Number of Epochs", pad=15)
    pyplot.savefig(model_name.replace(" ", "") + "_acc.png", dpi=300)
    pyplot.clf()


if __name__ == "__main__":
    print("\nRunning script ...\n")
    print("Loading MNIST Dataset ...\n")
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    print("Finished loading MNIST Dataset. \n")

    print("\nSGD with No Momentum\n")
    m_sgd1, h_sgd1 = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha, 0.0, batch_size, epochs)
    (test_loss_sgd1, test_acc_sgd1) = evaluate_model(Xs_te, Ys_te, m_sgd1)
    h_sgd1.history['test_loss'] = test_loss_sgd1
    h_sgd1.history['test_accuracy'] = test_acc_sgd1

    print("\nSGD with Momentum\n")
    m_sgd2, h_sgd2 = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, alpha, beta, batch_size, epochs)
    (test_loss_sgd2, test_acc_sgd2) = evaluate_model(Xs_te, Ys_te, m_sgd2)
    h_sgd2.history['test_loss'] = test_loss_sgd2
    h_sgd2.history['test_accuracy'] = test_acc_sgd2

    print("\nADAM\n")
    m_adam, h_adam = train_fully_connected_adam(Xs_tr, Ys_tr, d1, d2, alpha_adam, rho1, rho2, batch_size, epochs)
    (test_loss_adam, test_acc_adam) = evaluate_model(Xs_te, Ys_te, m_adam)
    h_adam.history['test_loss'] = test_loss_adam
    h_adam.history['test_accuracy'] = test_acc_adam

    print("\nSGD with Batch Normalization\n")
    m_bn, h_bn = train_fully_connected_bn_sgd(Xs_tr, Ys_tr, d1, d2, alpha, beta, batch_size, epochs)
    (test_loss_bn, test_acc_bn) = evaluate_model(Xs_te, Ys_te, m_bn)
    h_bn.history['test_loss'] = test_loss_bn
    h_bn.history['test_accuracy'] = test_acc_bn

    print("\nConvolutional Neural Network\n")
    m_cnn, h_cnn = train_CNN_sgd(Xs_tr, Ys_tr, alpha_adam, rho1, rho2, batch_size, epochs)
    (test_loss_cnn, test_acc_cnn) = evaluate_model(Xs_te, Ys_te, m_cnn)
    h_cnn.history['test_loss'] = test_loss_cnn
    h_cnn.history['test_accuracy'] = test_acc_cnn


    print("\nPlotting figures ...\n")  
    plot_graphs(h_sgd1, "SGD with No Momentum")
    plot_graphs(h_sgd2, "SGD with Momentum")
    plot_graphs(h_adam, "ADAM")
    plot_graphs(h_bn, "SGD with Batch Normalization")
    plot_graphs(h_cnn, "Convolutional Neural Network")
    print("Finished plotting figures. \n")
    print("Plots are saved as images in the current directory where the script is run.\n")


    print("MODEL STATISTICS\n")
    print("SGD with No Momentum")
    print(str(h_sgd1.history) + "\n")
    print("SGD with Momentum")
    print(str(h_sgd2.history) + "\n")
    print("ADAM")
    print(str(h_adam.history) + "\n")
    print("SGD with Batch Normalization")
    print(str(h_bn.history) + "\n")
    print("Convolutional Neural Network")
    print(str(h_cnn.history) + "\n")

    
    print("Grid Search - alpha values\n")
    a_lst = [1.0,0.3,0.1,0.03,0.01,0.003,0.001]
    
    val_acc = []
    for a in a_lst:
        m_sgda, h_sgda = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d2, a, beta, batch_size, epochs)
        print("\nalpha value : " + str(a))
        print("Validation accuracy : " + str(h_sgda.history['val_sparse_categorical_accuracy'][-1]))
        print("Validation loss : " + str(h_sgda.history['val_loss'][-1]) + "\n")
        val_acc.append(h_sgda.history['val_sparse_categorical_accuracy'][-1])
    
    ii = numpy.argmax(numpy.array(val_acc))
    print("Best alpha value(s) : \n")
    a_lst = numpy.array(a_lst)
    print(a_lst[ii])
 
    print("Grid Search\n")
    a_list = [0.01, 0.1, 1.0]
    b_list = [0.1, 0.5, 0.9]
    d2_list = [128, 256, 512]
    start_gs = time.time()
    for a in a_list:
        for b in b_list:
            for d in d2_list:
                m_gs, h_gs = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d, a, b, batch_size, epochs)
                print("\nalpha value : " + str(a) + " beta value : " + str(b) + " d2 value : " + str(d))
                print("Validation accuracy : " + str(h_gs.history['val_sparse_categorical_accuracy'][-1]))
                print("Validation loss : " + str(h_gs.history['val_loss'][-1]))
                (test_loss_gs, test_acc_gs) = evaluate_model(Xs_te, Ys_te, m_gs)
                print("Test accuracy : " + str(test_acc_gs))
                print("Test loss : " + str(test_loss_gs)+ "\n")
    end_gs = time.time()
    print("Grid Search Runtime : " + str(end_gs - start_gs) + "\n")
    
    print("Random Search\n")
    arv = loguniform(0.01, 1.0).rvs(size = 10)
    brv = uniform().rvs(size = 10)
    d2rv = random.randint(low = 128, high = 513, size = 10)

    start_rs = time.time()
    for i in range(10):
        a = arv[i]
        b = brv[i]
        d = d2rv[i]
        m_rs, h_rs = train_fully_connected_sgd(Xs_tr, Ys_tr, d1, d, a, b, batch_size, epochs)
        print("\nalpha value : " + str(a) + " beta value : " + str(b) + " d2 value : " + str(d))
        print("Validation accuracy : " + str(h_rs.history['val_sparse_categorical_accuracy'][-1]))
        print("Validation loss : " + str(h_rs.history['val_loss'][-1]))
        (test_loss_rs, test_acc_rs) = evaluate_model(Xs_te, Ys_te, m_rs)
        print("Test accuracy : " + str(test_acc_rs))
        print("Test loss : " + str(test_loss_rs)+ "\n")
    end_rs = time.time()
    print("Random Search Runtime : " + str(end_rs - start_rs) + "\n")
    
    print("Finished running script. \n")
