import torch.nn as nn
import torch
import torchvision.models as models

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            nn.BatchNorm1d(embed_size)
        )
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.BatchNorm1d(vocab_size)
        )

    def forward(self, features, captions):
        emb = self.embed[0](captions)
        B, T, E = emb.shape
        emb = self.embed[1](emb.view(-1, E)).view(B, T, E)

        inputs = torch.cat((features.unsqueeze(1), emb[:, :-1, :]), dim=1)
        out, _ = self.gru(inputs)
        logits = self.linear[0](out)
        logits = self.linear[1](logits.view(-1, logits.size(-1))).view(*logits.shape)
        return logits

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            nn.BatchNorm1d(embed_size)
        )
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.BatchNorm1d(vocab_size)
        )

    def forward(self, features, captions):
        emb = self.embed[0](captions)
        B, T, E = emb.shape
        emb = self.embed[1](emb.view(-1, E)).view(B, T, E)

        inputs = torch.cat((features.unsqueeze(1), emb[:, :-1, :]), dim=1)
        out, _ = self.lstm(inputs)
        logits = self.linear[0](out)
        logits = self.linear[1](logits.view(-1, logits.size(-1))).view(*logits.shape)
        return logits

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            nn.BatchNorm1d(embed_size)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Sequential(
            nn.Linear(embed_size, vocab_size),
            nn.BatchNorm1d(vocab_size)
        )

    def forward(self, features, captions):
        emb = self.embed[0](captions)
        B, T, E = emb.shape
        emb = self.embed[1](emb.view(-1, E)).view(B, T, E)

        inputs = torch.cat((features.unsqueeze(1), emb[:, :-1, :]), dim=1)
        out = self.transformer(inputs)
        logits = self.linear[0](out)
        logits = self.linear[1](logits.view(-1, logits.size(-1))).view(*logits.shape)
        return logits

class GRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.project = nn.Sequential(
            nn.Linear(2048, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU()
        )
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        feats = self.encoder(images).view(images.size(0), -1)
        projected_feats = self.project(feats)
        return self.decoder(projected_feats, captions)

class LSTMModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # resnet.load_state_dict(torch.load("pretrained_encoder.pth"))
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.project = nn.Sequential(
            nn.Linear(2048, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU()
        )
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        feats = self.encoder(images).view(images.size(0), -1)
        projected_feats = self.project(feats)
        return self.decoder(projected_feats, captions)

class TransformerModel(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=4, num_layers=2):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.project = nn.Sequential(
            nn.Linear(2048, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU()
        )
        self.decoder = DecoderTransformer(embed_size, vocab_size, num_heads, num_layers)

    def forward(self, images, captions):
        feats = self.encoder(images).view(images.size(0), -1)
        projected_feats = self.project(feats)
        return self.decoder(projected_feats, captions)
