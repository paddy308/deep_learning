# Libraries
import os
import random

import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
from config import NETWORK_TO_USE, DIRECTORY_MODELS


def main(neuralNet):
    # Get a random image from the MNIST test_set
    pictureIndex = random.randint(0, len(validation_data))
    if random.randint(0, 1):
        picture = validation_data[pictureIndex]
    else:
        picture = test_data[pictureIndex]

    guess, real = neuralNet.evaluate_single([picture])
    visualize(picture, guess)

    print("{real} / {guess}  -  {i}".format(real=real, guess=guess, i=pictureIndex))
    return real == guess


def visualize(pic, guess):
    pixels, label = pic
    pixels = np.dot(pixels, 256)
    pixels.tolist()
    pixels = np.array(pixels, dtype='uint8')
    pixels = np.invert(pixels)
    pixels = pixels.reshape((28, 28))

    plt.title('Label is {label}, Guess is {guess}'.format(label=label, guess=guess))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    pass


def load_network(file_path):
    if NETWORK_TO_USE == 1:
        import network as network
    elif NETWORK_TO_USE == 2:
        import network2 as network
    elif NETWORK_TO_USE == 3:
        # import network3 as network
        print("Visualization for Network3 isn't implemented yet.")
        exit(1)

    # Load the neural net
    return network.load(filename=file_path)

if __name__ == "__main__":
    # Load the network from the file.
    filepath = os.path.join(DIRECTORY_MODELS, "NeuralNetwork_{}.json".format(NETWORK_TO_USE))
    neural_network = load_network(file_path=filepath)

    # We don't want to test our neural net on trainings data
    validation_data, test_data = mnist_loader.load_data_wrapper()[1:]

    print("Real / Guess")
    for _ in range(5):
        # Plot a image and let the neural network classify it.
        main(neuralNet=neural_network)
