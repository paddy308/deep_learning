import os

import mnist_loader
from config import DIRECTORY_MODELS
from config import LOAD_NEURAL_NETWORK, SAVE_NEURAL_NETWORK
from config import EPOCHS, LAYERS, LEARNING_RATE, MINI_BATCH_SIZE, NETWORK_TO_USE, NETWORK_TO_USE


def run():
    neuralNet = None
    neuralNetworkFile = os.path.join(DIRECTORY_MODELS, "NeuralNetwork_{}.json".format(NETWORK_TO_USE))

    # Load and Split the data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    if LOAD_NEURAL_NETWORK:
        print("Load the Neural Network from \"{}\"".format(neuralNetworkFile))
        neuralNet = load(filepath=neuralNetworkFile)

    # USE THIS FOR network.py
    if NETWORK_TO_USE == 1:
        # If you don't load a neural network
        # then it will construct a new one with the given layers
        if not neuralNet:
            neuralNet = network.Network(LAYERS)

        # Train the neural net
        neuralNet.SGD(training_data, EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE, test_data=test_data)

    # USE THIS FOR network2.py
    elif NETWORK_TO_USE == 2:
        # If you don't load a neural network
        # then it will construct a new one with the given layers
        if not neuralNet:
            neuralNet = network.Network(LAYERS)

        # Train the neural net
        neuralNet.SGD(training_data, EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE, lmbda=5.0,
                      evaluation_data=validation_data, monitor_evaluation_accuracy=True)

    # Use this for network3.py
    # network3.py needs the package Theano
    # If you own a Nvidia GPU you can run this code on your GPU
    # Otherwise this code will be executed by your CPU
    elif NETWORK_TO_USE == 3:

        # If you don't load an existing neural net
        # then you can choose the neural net structure here
        # networkToUse:
        #   1: One fully connected and one softmax layer
        #   2: Same as (1) but there is a convolutional layer added
        #   3: Same as (2) but there is a second convolutional layer added
        if not neuralNet:
            neuralNet = construct_initial_neural_network(networkStructure=3)

        # Load the data in a way that Theano can copy the GPU
        training_data, validation_data, test_data = network.load_data_shared()

        # Train the neural net
        neuralNet.SGD(training_data, EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE,
                      validation_data, test_data)

    if SAVE_NEURAL_NETWORK:
        print("Save the Neural Network to \"{}\"".format(neuralNetworkFile))
        neuralNet.save(filename=neuralNetworkFile)


def load(filepath):
    try:
        loadedNetwork = network.load(filename=filepath)
    except Exception as e:
        print(e)
        loadedNetwork = None

    return loadedNetwork


def construct_initial_neural_network(networkStructure=1):
    """Choose which network will be used.
    :rtype: Network object
    """
    from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
    from network3 import ReLU

    if networkStructure == 2:
        # This network has one Convolutional Layer
        neuralNet = network.Network([
            ConvPoolLayer(image_shape=(MINI_BATCH_SIZE, 1, 28, 28),
                          filter_shape=(20, 1, 5, 5),
                          poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], MINI_BATCH_SIZE)

    elif networkStructure == 3:
        # This network has two Convolutional Layers
        neuralNet = network.Network([
            ConvPoolLayer(image_shape=(MINI_BATCH_SIZE, 1, 28, 28),
                          filter_shape=(20, 1, 5, 5),
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(MINI_BATCH_SIZE, 20, 12, 12),
                          filter_shape=(40, 20, 5, 5),
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=40 * 4 * 4, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], MINI_BATCH_SIZE)
    else:
        # Default network without any Convolutional Layers
        neuralNet = network.Network([
            FullyConnectedLayer(n_in=784, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], MINI_BATCH_SIZE)

    return neuralNet


if __name__ == "__main__":
    if NETWORK_TO_USE == 1:
        import network as network
    elif NETWORK_TO_USE == 2:
        import network2 as network
    elif NETWORK_TO_USE == 3:
        import network3 as network

    run()
