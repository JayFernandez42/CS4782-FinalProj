#@title Setup & Imports
import os
import zipfile
import pandas as pd
# from src.MAE.MAE_datas1et import MAEDataset
# from src.MAE.MAE_model import MaskedAutoencoderCNN
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path




from PIL import Image
import json
# import nltk
# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import io
import random
import config
import dataset
import utils
import eval
import model
from config import *
from dataset import load_csv_paths, create_dataloaders
from utils import build_vocab
from model import GRNN
from train import train_model, plot_history
from eval import test_loss, generate_question
from dataset import VQGTensorDataset
from model import TransformerModel
import torch.nn as nn
import torch.optim as optim




import nltk
import os

# nltk.download('punkt', download_dir='~/nltk_data')
# nltk.data.path.append(os.path.expanduser('~/nltk_data'))

# nltk.download('punkt_tab')





# Assuming your trained Transformer model is named 'transformer_model'

def main(EMBED_SIZE = 256,
    HIDDEN_SIZE = 512,
    NUM_LAYERS = 1,
    LR = 0.001,
    EPOCHS = 7,
    MAX_LENGTH = 20,
    BATCH_SIZE = 64):
    # All your current code here
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1
    LR = 0.001
    EPOCHS = 7
    MAX_LENGTH = 20
    BATCH_SIZE = 64

    # gqa_test_tidx = pd.read_csv("/Users/jacobfernandez/Desktop/CornellMEngSP25/DeepLearning/FinalProject/github/data/gqa_test_tensor_index.csv")
    # gqa_train_tidx   = pd.read_csv("/Users/jacobfernandez/Desktop/CornellMEngSP25/DeepLearning/FinalProject/github/data/gqa_train_tensor_index.csv")
    # gqa_val_tidx  = pd.read_csv("/Users/jacobfernandez/Desktop/CornellMEngSP25/DeepLearning/FinalProject/github/data/gqa_val_tensor_index.csv")

    train_df = pd.read_csv('/Users/jacobfernandez/Desktop/CornellMEngSP25/DeepLearning/FinalProject/github/train_split.csv')
    val_df   = pd.read_csv('/Users/jacobfernandez/Desktop/CornellMEngSP25/DeepLearning/FinalProject/github/val_split.csv')
    test_df  = pd.read_csv('/Users/jacobfernandez/Desktop/CornellMEngSP25/DeepLearning/FinalProject/github/test_split.csv')

    # 1. Concatenate Datasets
    # combined_df = pd.concat([train_df, val_df, test_df])

    # # 2. Sample Unique Images (based on 'tensor_path')
    # unique_image_ids = combined_df['tensor_path'].unique()  # Get unique image IDs
    # subset_image_ids = np.random.choice(unique_image_ids, size=10000, replace=False)  # Sample 10k unique IDs
    # subset_df = combined_df[combined_df['tensor_path'].isin(subset_image_ids)]  # Filter DataFrame

    # # 3. Train-Validation-Test Split
    # train_df, temp_df = train_test_split(subset_df, test_size=0.3, random_state=42)
    # val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


    questions = pd.concat([train_df, val_df, test_df])["questions"].dropna().tolist()
    vocab, idx_to_word = build_vocab(questions)

    train_dataset = VQGTensorDataset(train_df, vocab, MAX_LENGTH)
    val_dataset   = VQGTensorDataset(val_df, vocab, MAX_LENGTH)
    test_dataset  = VQGTensorDataset(test_df, vocab, MAX_LENGTH)


    # Dataloaders
    from torch.utils.data import DataLoader
    dataloaders = {
        "gqa": {
            "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8),
            "val":   DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers =8),
            "test":  DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers =8),
        }
    }

    # from model import TransformerModel  # assuming you've imported from your updated model.py

    transformer_model = TransformerModel(EMBED_SIZE, len(vocab),num_layers=NUM_LAYERS, use_resnet = False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=LR, weight_decay=1e-5)
    history_transformer = train_model(
        transformer_model,
        dataloaders["gqa"]["train"],
        dataloaders["gqa"]["val"],
        vocab,
        criterion,
        optimizer,
        device,
        5
    )
    plot_history(history_transformer)
    test_loss(transformer_model, dataloaders["gqa"]["test"], vocab, criterion, device)

    save_dir = Path("models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Save the model
    torch.save(transformer_model.state_dict(), save_dir / "transformer_model.pth")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # or "fork" on Linux
    main(EMBED_SIZE = 256,
    HIDDEN_SIZE = 512,
    NUM_LAYERS = 4,
    LR = 0.001,
    EPOCHS = 7,
    MAX_LENGTH = 20,
    BATCH_SIZE = 128)
