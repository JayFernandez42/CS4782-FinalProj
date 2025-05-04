# mae_train.py
import torch
from torch.utils.data import DataLoader
from mae_dataset import MAEDataset
from mae_model import MaskedAutoencoderCNN
import os
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

# --- CONFIG ---
EPOCHS = 15
BATCH_SIZE = 64
PATCH_SIZE = 16
MASK_RATIO = 0.75
EMBED_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load paths from all three splits
csvs = ["data/bing_data/bing_train_tensor_index.csv", "data/bing_data/bing_val_tensor_index.csv", "data/bing_data/bing_test_tensor_index.csv"]
paths = []
for csv in csvs:
    df = pd.read_csv(csv)
    paths += [os.path.join("data/bing_data", p) for p in df["tensor_path"].dropna().tolist()]

# Create dataset and dataloader
dataset = MAEDataset(paths, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = MaskedAutoencoderCNN(embed_dim=EMBED_DIM, patch_size=PATCH_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for masked, target, mask in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        masked, target, mask = masked.to(DEVICE), target.to(DEVICE), mask.to(DEVICE)
        output = model(masked, mask)
        loss = F.mse_loss(output[mask], target[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"🧪 Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

# Save encoder weights
torch.save(model.encoder.state_dict(), "pretrained_encoder.pth")
print("✅ Saved encoder weights to pretrained_encoder.pth")
