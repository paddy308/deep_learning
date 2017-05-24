import os


# PARAM
NETWORK_TO_USE = 2
LOAD_NEURAL_NETWORK = False
SAVE_NEURAL_NETWORK = True

# LAYERS = [784, 200, 100, 26]          # Number of neurons per layer
LAYERS = [784, 30, 10]          # Number of neurons per layer
EPOCHS = 30                     # Number of training epochs
MINI_BATCH_SIZE = 10            # Size of the mini batch
LEARNING_RATE = 0.1             # Learning rate


# MNIST CREATOR
FOLDER_NEW_DATA = "bilder"
TRAINING_SIZE = 0.7     # In percent
VALIDATION_SIZE = 0.15  # In percent
TEST_SIZE = 0.15        # In percent


# DIRECTORIES
DIRECTORY_ROOT = os.path.dirname(__file__)
DIRECTORY_CONFIG = os.path.dirname(__file__)
DIRECTORY_DATA = os.path.join(DIRECTORY_ROOT, 'data')
DIRECTORY_NEW_DATA = os.path.join(DIRECTORY_DATA, FOLDER_NEW_DATA)

# NUMBERS
DIRECTORY_MODELS = os.path.join(DIRECTORY_ROOT, 'saved', 'numbers')
FILE_DATA = os.path.join(DIRECTORY_DATA, "mnist.pkl.gz")

# CHARACTERS
# DIRECTORY_MODELS = os.path.join(DIRECTORY_ROOT, 'saved', 'chars')
# FILE_DATA = os.path.join(DIRECTORY_DATA, "bilder.pkl.gz")
