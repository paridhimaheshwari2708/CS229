import os
import sys
import clip
import torch
import argparse
import numpy as np
from tqdm import tqdm
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

sys.path.append("Models")
from data import Memes
from vqa import ImageEmbedding, word_embedding_layer

DATA_PATH = "/dfs/user/paridhi/CS229/Project/Data"
VOCABULARY_PATH = os.path.join(DATA_PATH, "vocabulary.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(torch.nn.Module):

	def __init__(self, image_mode, text_mode):
		super(Encoder, self).__init__()

		self.image_mode = image_mode
		self.text_mode = text_mode
		
		self.image_channel = ImageEmbedding(image_channel_type="I", output_size=1024, extract_features=True, mode=self.image_mode)

		self.vocabulary = np.load(VOCABULARY_PATH)
		if self.text_mode == 'clip':
			self.clip, _ = clip.load("ViT-B/32", device=device)
			self.unk = np.where(self.vocabulary == 'unk')[0][0]
		else:
			self.word_embeddings = word_embedding_layer(self.vocabulary, mode=self.text_mode)

	def forward(self, image, text):
		image_embed = self.image_channel(image)
		if self.text_mode == 'clip':
			phrases = []
			for row in text:
				phrases.append(' '.join([self.vocabulary[idx] for idx in row if idx != self.unk]))
			text = clip.tokenize(phrases).to(device)
			text_embed = self.clip.encode_text(text)
		else:
			text_embed = self.word_embeddings(text).mean(dim=1)
		return image_embed, text_embed

def scatter_plot(data, label, save):
	plt.figure()
	data1 = data[label==1]
	data2 = data[label==0]
	plt.scatter(data1[:,0], data1[:,1])
	plt.scatter(data2[:,0], data2[:,1])
	plt.legend(["Misogynous", "Not Misogynous"], prop={'size': 15})
	plt.show()
	plt.savefig(save)

parser = argparse.ArgumentParser()
parser.add_argument("--image_mode", action="store", type=str, choices=["general", "clip"], required=True)
parser.add_argument("--text_mode", action="store", type=str, choices=["glove", "urban", "clip"], required=True)
args = parser.parse_args()

image_mode, text_mode = args.image_mode, args.text_mode

model = Encoder(image_mode=image_mode, text_mode=text_mode).to(device)
model.eval()

labels, image_embeddings, text_embeddings = [], [], []
# for subset in ["train", "val", "test"]:
for subset in ["test"]:
	loader = DataLoader(Memes(subset, mode="TaskB", image_mode=image_mode), shuffle=False, batch_size=128)
	for (image, text, label) in tqdm(loader):
		image, text = image.to(device), text.to(device)
		image_embed, text_embed = model(image, text)
		image_embeddings.append(image_embed.cpu().detach().numpy())
		text_embeddings.append(text_embed.cpu().detach().numpy())
		labels.append(label.detach().numpy())
labels, image_embeddings, text_embeddings = np.concatenate(labels), np.concatenate(image_embeddings), np.concatenate(text_embeddings)
combined_embeddings = np.concatenate((image_embeddings, text_embeddings), axis=1)

# # Image
# tsne_image = TSNE(n_components=2, n_jobs=4, verbose=2).fit_transform(image_embeddings)
# scatter_plot(tsne_image, labels[:,0], "image_tsne_{}.png".format(image_mode))

# pca_image = PCA(n_components=2).fit_transform(image_embeddings)
# scatter_plot(pca_image, labels[:,0], "image_pca_{}.png".format(image_mode))

# umap_image = UMAP(n_components=2, init='random', random_state=0).fit_transform(image_embeddings)
# scatter_plot(umap_image, labels[:,0], "image_umap_{}.png".format(image_mode))

# # Text
# tsne_text = TSNE(n_components=2, n_jobs=4, verbose=2).fit_transform(text_embeddings)
# scatter_plot(tsne_text, labels[:,0], "text_tsne_{}.png".format(text_mode))

# pca_text = PCA(n_components=2).fit_transform(text_embeddings)
# scatter_plot(pca_text, labels[:,0], "text_pca_{}.png".format(text_mode))

# umap_text = UMAP(n_components=2, init='random', random_state=0).fit_transform(text_embeddings)
# scatter_plot(umap_text, labels[:,0], "text_umap_{}.png".format(text_mode))

# Combined
tsne_combined = TSNE(n_components=2, n_jobs=4, verbose=2).fit_transform(combined_embeddings)
scatter_plot(tsne_combined, labels[:,0], "combined_tsne_{}_{}.png".format(image_mode, text_mode))

pca_combined = PCA(n_components=2).fit_transform(combined_embeddings)
scatter_plot(pca_combined, labels[:,0], "combined_pca_{}_{}.png".format(image_mode, text_mode))

umap_combined = UMAP(n_components=2, init='random', random_state=0).fit_transform(combined_embeddings)
scatter_plot(umap_combined, labels[:,0], "combined_umap_{}_{}.png".format(image_mode, text_mode))
