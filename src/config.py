# config.py
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
LR = 0.001
EPOCHS = 10
MAX_LENGTH = 20
BATCH_SIZE = 64
