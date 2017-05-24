from config import DIRECTORY_NEW_DATA, TEST_SIZE, TRAINING_SIZE, VALIDATION_SIZE

# Standard library
import os
import gzip
import pickle
from PIL import Image
from random import shuffle

# Third-party libraries
import numpy as np


def get_label(char):
    """Change character to a number between 0 and 25.
    A = 0
    B = 1
    ...
    Y = 24
    Z = 25"""
    value = ord(char) - 65
    return value


def load_data():
    # All data
    data = []
    label = []
    # Splitted data
    te_data = []
    te_label = []
    tr_data = []
    tr_label = []
    va_data = []
    va_label = []

    distributionList = []
    filecounter = 0

    for directory in os.listdir(DIRECTORY_NEW_DATA):
        charDirectory = os.path.join(DIRECTORY_NEW_DATA, directory)

        for filename in os.listdir(charDirectory):
            # This list will be shuffled, then you can distribute the data to the three lists.
            distributionList.append(filecounter)

            # Extract the character from the filename
            character = filename[:1].capitalize()

            # Get the character as a number between 0 and 25
            label.append(get_label(char=character))

            # Open the image
            filepath = os.path.join(charDirectory, filename)
            with Image.open(filepath) as im:
                # Convert the pixels in a numpy array
                # Normalize the pixels that the values are between 0 and 1
                im = np.asarray(im) / 255

                # Reshape the array from 28x28 to 784x1
                data.append(np.reshape(im, (784, )))

            filecounter += 1

    shuffle(distributionList)
    for enumcount, number in enumerate(distributionList):
        if enumcount < len(distributionList) * VALIDATION_SIZE:
            va_data.append(data[number])
            va_label.append(label[number])
        elif enumcount >= len(distributionList) * (1 - TEST_SIZE):
            te_data.append(data[number])
            te_label.append(label[number])
        else:
            tr_data.append(data[number])
            tr_label.append(label[number])

    training_data = (np.asarray(tr_data), np.asarray(tr_label))
    validation_data = (np.asarray(va_data), np.asarray(va_label))
    test_data = (np.asarray(te_data), np.asarray(te_label))

    pklgzFile = save_as_pkl_gz(tr_d=training_data, va_d=validation_data, te_d=test_data)
    print("Your data was stored at: '{filepath}'".format(filepath=pklgzFile))


def save_as_pkl_gz(tr_d, va_d, te_d):
    filename = DIRECTORY_NEW_DATA + ".pkl.gz"
    with gzip.open(filename, "wb") as file:
        pickle.dump([tr_d, va_d, te_d], file)

    return filename


if __name__ == '__main__':
    load_data()
