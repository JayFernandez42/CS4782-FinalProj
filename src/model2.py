import torch.nn as nn
import torch

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            nn.BatchNorm1d(embed_size)  # ✅ normalize embedding output
        )
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.BatchNorm1d(vocab_size)  # ✅ normalize logits before softmax
        )

    def forward(self, features, captions):
        embeddings = self.embed(captions)  # shape: [B, T, E] → [B, E]
        if embeddings.dim() == 3:
            B, T, E = embeddings.shape
            embeddings = embeddings.view(-1, E)
            embeddings = self.embed[1](embeddings)  # Apply BN1d
            embeddings = embeddings.view(B, T, E)
        
        inputs = torch.cat((features.unsqueeze(1), embeddings[:, :-1, :]), dim=1)
        out, _ = self.gru(inputs)
        out = self.linear[0](out)
        B, T, V = out.shape
        out = out.view(-1, V)
        out = self.linear[1](out)
        out = out.view(B, T, V)
        return out

class GRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),               # ✅ normalize conv output
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, embed_size),
            nn.BatchNorm1d(embed_size),       # ✅ normalize embedding vector
            nn.ReLU()
        )
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        feats = self.encoder(images)
        return self.decoder(feats, captions)
