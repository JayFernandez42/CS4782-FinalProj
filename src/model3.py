# model.py
import torch.nn as nn
import torch
import torchvision.models as models

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings[:, :-1, :]), dim=1)
        out, _ = self.gru(inputs)
        return self.linear(out)

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings[:, :-1, :]), dim=1)
        out, _ = self.lstm(inputs)
        return self.linear(out)

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings[:, :-1, :]), dim=1)  # [B, T, D]
        out = self.transformer(inputs.transpose(0, 1)).transpose(0, 1)  # [T, B, D] → [B, T, D]
        return self.linear(out)

class GRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.project = nn.Linear(2048, embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        feats = self.encoder(images)
        feats = feats.view(feats.size(0), -1)
        projected_feats = self.project(feats)
        return self.decoder(projected_feats, captions)

class LSTMModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.project = nn.Linear(2048, embed_size)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        feats = self.encoder(images)
        feats = feats.view(feats.size(0), -1)
        projected_feats = self.project(feats)
        return self.decoder(projected_feats, captions)

class TransformerModel(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=4, num_layers=2):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.project = nn.Linear(2048, embed_size)
        self.decoder = DecoderTransformer(embed_size, vocab_size, num_heads, num_layers)

    def forward(self, images, captions):
        feats = self.encoder(images)
        feats = feats.view(feats.size(0), -1)
        projected_feats = self.project(feats)
        return self.decoder(projected_feats, captions)
