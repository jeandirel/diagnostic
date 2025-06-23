import os

# Define base project directory
BASE_DIR = r"C:/Users/asmag/OneDrive/Documents/AI_FOR_HEALTH"

# Dataset paths
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")

# Annotation files
TRAIN_ANNOTATIONS = os.path.join(TRAIN_DIR, "_annotations.coco.json")
VALID_ANNOTATIONS = os.path.join(VALID_DIR, "_annotations.coco.json")

# Other configurations
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)  # Resize images to this size for training
NUM_WORKERS = 4  # Number of workers for data loading
NUM_EPOCHS = 10  # Number of epochs for training
LEARNING_RATE = 0.001  # Learning rate for the optimizer
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "unet.pth")
LOG_DIR = os.path.join(BASE_DIR, "logs")  # Directory for saving logs
os.makedirs(LOG_DIR, exist_ok=True)  # Create log directory if it doesn't exist
