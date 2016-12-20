import mnist_loader
import pickle
import os


def run():
    # DEFINES WHICH NETWORK WILL BE USED
    networkToUse = 3
    loadNeuralNetwork = True
    saveNeuralNetwork = True
    neuralNetworkDump = os.path.join(os.getcwd(), "saves")
    neuralNetworkDump = os.path.join(neuralNetworkDump, "NeuralNetwork_{}.nnd".format(networkToUse))

    # PARAMETER
    layers = [784, 30, 10]
    epochs = 3
    mini_batch_size = 10
    learning_rate = 0.1

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    try:
        with open(neuralNetworkDump, "rb") as file:
            net = pickle.load(file)
        print("Neural Network loaded from \"{}\"".format(neuralNetworkDump))
    except IOError as e:
        print(e)
        loadNeuralNetwork = False
        net = None

    # USE THIS FOR network.py
    if networkToUse == 1:
        import network

        if not loadNeuralNetwork:
            net = network.Network(layers)

        net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)

    # USE THIS FOR network2.py
    elif networkToUse == 2:
        import network2

        if not loadNeuralNetwork:
            net = network2.Network(layers)

        net.SGD(training_data, epochs, mini_batch_size, learning_rate, lmbda=5.0,
                evaluation_data=validation_data, monitor_evaluation_accuracy=True)

    # USE THIS FOR network3.py
    elif networkToUse == 3:
        import network3
        from network3 import Network
        from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

        training_data, validation_data, test_data = network3.load_data_shared()

        if not loadNeuralNetwork:
            #net = Network([
            #    FullyConnectedLayer(n_in=784, n_out=100),
            #    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                              filter_shape=(20, 1, 5, 5),
                              poolsize=(2, 2)),
                FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

        net.SGD(training_data, 60, mini_batch_size, 0.1,
                validation_data, test_data)

    if saveNeuralNetwork:
        print("Save the Neural Network to \"{}\"".format(neuralNetworkDump))
        with open(neuralNetworkDump, "wb") as file:
            pickle.dump(net, file)


if __name__ == "__main__":
    run()
