import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import VOCABULARY_PATH, GLOVE_PATH, GLOVE_EMBEDDING_SIZE, GLOVE_OOV_SCALE


def load_glove():
    glove = {}
    with open(GLOVE_PATH, "rb") as f:
        for line in f:
            line = line.decode().split()
            word = line[0]
            vec = np.array(line[1:]).astype(np.float)
            glove[word] = vec
    return glove


def word_embedding_layer(vocabulary, trainable=True):
    words_in_glove = 0
    glove = load_glove()
    weight_matrix = np.zeros((len(vocabulary), GLOVE_EMBEDDING_SIZE))
    for i, word in enumerate(vocabulary):
        try:
            weight_matrix[i] = glove[word]
            words_in_glove += 1
        except KeyError:
            # TODO: same random initialization for out-of-vocab words from test set
            weight_matrix[i] = np.random.normal(
                scale=GLOVE_OOV_SCALE, size=(GLOVE_EMBEDDING_SIZE,)
            )
    print("# of words: {}".format(len(vocabulary)))
    print("# of words found: {}".format(words_in_glove))

    embedding_layer = nn.Embedding(len(vocabulary), GLOVE_EMBEDDING_SIZE)
    embedding_layer.load_state_dict({"weight": torch.tensor(weight_matrix)})
    if not trainable:
        embedding_layer.weight.requires_grad = False
    return embedding_layer


class ImageEmbedding(nn.Module):
    def __init__(self, output_size=1024, extract_features=False):
        super(ImageEmbedding, self).__init__()

        self.cnn = models.vgg16(pretrained=True).features
        # self.cnn = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())) #FIXME: Only temporary for the first experiment

        for param in self.cnn.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(nn.Linear(512, output_size), nn.Tanh())

        self.extract_features = extract_features

    def forward(self, image):
        if self.extract_features:
            # N * 224 * 224 -> N * 512 * 7 * 7
            image = self.cnn(image)
            # N * 512 * 7 * 7 -> N * 512 * 49 -> N * 49 * 512
            image = image.view(-1, 512, 49).transpose(1, 2)
        # N * 49 * 512 -> N * 49 * 1024
        image_embedding = self.fc(image)
        return image_embedding


class TextEmbedding(nn.Module):
    def __init__(self, input_size=500, output_size=1024, batch_first=True):
        super(TextEmbedding, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=output_size, batch_first=batch_first
        )

    def forward(self, text):
        # seq_len * N * 500 -> (1 * N * 1024, 1 * N * 1024)
        _, hx = self.lstm(text)
        # (1 * N * 1024, 1 * N * 1024) -> 1 * N * 1024
        h, _ = hx
        text_embedding = h[0]
        return text_embedding


class Attention(nn.Module):
    def __init__(self, d=1024, k=512, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_text = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        # N * 49 * 1024 -> N * 49 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_text(vq).unsqueeze(dim=1)
        # N * 49 * 512
        ha = F.tanh(hi + hq)
        if getattr(self, "dropout"):
            ha = self.dropout(ha)
        # N * 49 * 512 -> N * 49 * 1 -> N * 49
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha)
        # (N * 49 * 1, N * 49 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u


class SANModel(nn.Module):
    def __init__(
        self,
        output_size,
        emb_size=1024,
        att_ff_size=512,
        num_att_layers=1,
        extract_img_features=True,
    ):
        super(SANModel, self).__init__()

        self.word_emb_size = GLOVE_EMBEDDING_SIZE
        self.image_channel = ImageEmbedding(
            output_size=emb_size, extract_features=extract_img_features
        )
        self.vocabulary = np.load(VOCABULARY_PATH)

        # NOTE the padding_idx below.
        self.word_embeddings = word_embedding_layer(self.vocabulary)
        self.text_channel = TextEmbedding(
            input_size=self.word_emb_size, output_size=emb_size
        )

        self.san = nn.ModuleList(
            [Attention(d=emb_size, k=att_ff_size)] * num_att_layers
        )

        self.mlp = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(emb_size, output_size))

    def forward(self, images, texts):
        image_embeddings = self.image_channel(images)
        embeds = self.word_embeddings(texts)

        text_embeddings = self.text_channel(embeds)
        vi = image_embeddings
        u = text_embeddings
        for att_layer in self.san:
            u = att_layer(vi, u)
        output = self.mlp(u)
        return output

    def saveCheckpoint(
        self, savePath, epoch, optimizer, bestTrainLoss, bestValLoss, isBest
    ):
        ckpt = {}
        ckpt["state"] = self.state_dict()
        ckpt["epoch"] = epoch
        ckpt["optimizer_state"] = optimizer.state_dict()
        ckpt["bestTrainLoss"] = bestTrainLoss
        ckpt["bestValLoss"] = bestValLoss
        torch.save(ckpt, os.path.join(savePath, "model.ckpt"))
        if isBest:
            torch.save(ckpt, os.path.join(savePath, "bestModel.ckpt"))

    def loadCheckpoint(self, loadPath, optimizer, loadBest=False):
        if loadBest:
            ckpt = torch.load(os.path.join(loadPath, "bestModel.ckpt"))
        else:
            ckpt = torch.load(os.path.join(loadPath, "model.ckpt"))
        self.load_state_dict(ckpt["state"])
        epoch = ckpt["epoch"]
        bestTrainLoss = ckpt["bestTrainLoss"]
        bestValLoss = ckpt["bestValLoss"]
        optimizer.load_state_dict(ckpt["optimizer_state"])
        return epoch, bestTrainLoss, bestValLoss
