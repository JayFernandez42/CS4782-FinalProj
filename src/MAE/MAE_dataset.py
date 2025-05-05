import torch
from torch.utils.data import Dataset
import random

class MAEDataset(Dataset):
    def __init__(self, tensor_paths, patch_size=16, mask_ratio=0.5):
        self.tensor_paths = tensor_paths
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        image = torch.load(self.tensor_paths[idx]).float()  # shape: [3, 224, 224]

        # Ensure divisible by patch size
        assert image.shape[1] % self.patch_size == 0 and image.shape[2] % self.patch_size == 0

        # Create patches
        patches = image.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, self.patch_size, self.patch_size)  # [N_patches, C, H, W]

        num_patches = patches.size(0)
        num_mask = int(self.mask_ratio * num_patches)

        mask_indices = random.sample(range(num_patches), num_mask)
        mask = torch.zeros(num_patches, dtype=torch.bool)
        mask[mask_indices] = True

        # Create masked version
        masked_patches = patches.clone()
        masked_patches[mask] = 0

        return masked_patches, patches, mask  # input, target, mask