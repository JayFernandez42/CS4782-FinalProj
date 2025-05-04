import torch
import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embed_dim)
        )

    def forward(self, x):
        B, N, C, H, W = x.shape  # x: [B, N_patches, 3, 16, 16]
        x = x.view(B * N, C, H, W)
        out = self.encoder(x)  # [B * N, embed_dim]
        return out.view(B, N, -1)

class DecoderMLP(nn.Module):
    def __init__(self, embed_dim=256, patch_size=16):
        super().__init__()
        self.patch_dim = 3 * patch_size * patch_size
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.patch_dim)
        )
        self.patch_size = patch_size

    def forward(self, x):
        B, N, D = x.shape
        out = self.decoder(x.view(B * N, D))
        return out.view(B, N, 3, self.patch_size, self.patch_size)

class MaskedAutoencoderCNN(nn.Module):
    def __init__(self, embed_dim=256, patch_size=16):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderMLP(embed_dim, patch_size)

    def forward(self, x, mask):
        encoded = self.encoder(x)  # [B, N, D]
        reconstructed = self.decoder(encoded)  # [B, N, 3, p, p]
        return reconstructed