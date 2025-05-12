# config.py
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_SIZE = 1024
HIDDEN_SIZE = 2048
NUM_LAYERS = 6
LR = 0.001
EPOCHS = 25
MAX_LENGTH = 10
BATCH_SIZE = 32
