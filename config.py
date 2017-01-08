import os


# DIRECTORIES
DIRECTORY_ROOT = os.path.dirname(__file__)
DIRECTORY_CONFIG = os.path.dirname(__file__)
DIRECTORY_DATA = os.path.join(DIRECTORY_ROOT, 'data')
DIRECTORY_MODELS = os.path.join(DIRECTORY_ROOT, 'saved')


# PARAM
NETWORK_TO_USE = 2
LOAD_NEURAL_NETWORK = False
SAVE_NEURAL_NETWORK = True

LAYERS = [784, 30, 10]          # Number of neurons per layer
EPOCHS = 30                     # Number of training epochs
MINI_BATCH_SIZE = 10            # Size of the mini batch
LEARNING_RATE = 0.1             # Learning rate
