import torch.nn as nn
import torch
import torchvision.models as models
import clip  # Add this import at the top of your file


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


class ClipTransformerModel(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=4, num_layers=2, use_clip=True):
        super().__init__()
        self.use_clip = use_clip

        self.img_fc = nn.Linear(512, embed_size)  # CLIP ViT-B/16 has 512-dim output

        if self.use_clip:
            # Load CLIP's ViT-B/16 encoder
            self.clip_model, _ = clip.load("ViT-B/16", device="cpu")  # Move to GPU in forward
            for param in self.clip_model.visual.parameters():
                param.requires_grad = False
        else:
            # Fallback if using direct 2048-dim features
            self.project = nn.Sequential(
                nn.Linear(2048, embed_size),
                nn.BatchNorm1d(embed_size),
                nn.ReLU()
            )

        self.decoder = DecoderTransformer(embed_size, vocab_size, num_heads, num_layers)

    def forward(self, images_or_feats, captions):
        if self.use_clip:
            image_features = self.clip_model.encode_image(images_or_feats).float()  # CLIP expects normalized image tensors
        else:
            image_features = images_or_feats  # 2048-dim precomputed .pt

        projected_feats = self.img_fc(image_features)
        return self.decoder(projected_feats, captions)




# class TransformerModel(nn.Module):
#     def __init__(self, embed_size, vocab_size, num_heads=4, num_layers=2):
#         super().__init__()
#         resnet = models.resnet50(pretrained=True)
#         self.encoder = nn.Sequential(*list(resnet.children())[:-1])
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#         self.project = nn.Sequential(
#             nn.Linear(2048, embed_size),
#             nn.BatchNorm1d(embed_size),
#             nn.ReLU()
#         )
#         self.decoder = DecoderTransformer(embed_size, vocab_size, num_heads, num_layers)

#     def forward(self, images, captions):
#         feats = self.encoder(images).view(images.size(0), -1)
#         projected_feats = self.project(feats)
#         return self.decoder(projected_feats, captions)

class TransformerModel(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=4, num_layers=2, use_resnet=True):
        super().__init__()
        self.use_resnet = use_resnet

        self.img_fc = nn.Linear(2048, embed_size)
        


        if self.use_resnet:
            resnet = models.resnet50(pretrained=True)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.project = nn.Sequential(
                nn.Linear(2048, embed_size),
                nn.BatchNorm1d(embed_size),
                nn.ReLU()
            )
        else:
            # Directly project the 2048-dim precomputed tensor
            self.project = nn.Sequential(
                nn.Linear(2048, embed_size),
                nn.BatchNorm1d(embed_size),
                nn.ReLU()
            )

        self.decoder = DecoderTransformer(embed_size, vocab_size, num_heads, num_layers)

    def forward(self, images_or_feats, captions):
        if self.use_resnet:
            feats = self.encoder(images_or_feats).view(images_or_feats.size(0), -1)
        else:
            feats = images_or_feats  # already 2048-dim features
        projected_feats = self.project(feats)
        return self.decoder(projected_feats, captions)

