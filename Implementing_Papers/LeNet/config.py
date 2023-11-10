from torch import cuda

# Training hyperparameters
NUM_CLASSES = 10
LEARNING_RATE = 0.1
BATCH_SIZE = 128
NUM_EPOCHS = 1

# Dataset
DATA_DIR = "data/"
NUM_WORKERS = 0

# Compute
SEED = 13567
ACCELERATOR = "cuda" if cuda.is_available() else "cpu"