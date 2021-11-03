import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import VOCABULARY_PATH, GLOVE_PATH, GLOVE_EMBEDDING_SIZE, GLOVE_OOV_SCALE

def load_glove():
	glove = {}
	with open(GLOVE_PATH, 'rb') as f:
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
			weight_matrix[i] = np.random.normal(scale = GLOVE_OOV_SCALE, size = (GLOVE_EMBEDDING_SIZE, ))
	print('# of words: {}'.format(len(vocabulary)))
	print('# of words found: {}'.format(words_in_glove))

	embedding_layer = nn.Embedding(len(vocabulary), GLOVE_EMBEDDING_SIZE)
	embedding_layer.load_state_dict({'weight': torch.tensor(weight_matrix)})
	if not trainable:
		embedding_layer.weight.requires_grad = False
	return embedding_layer

class MutanFusion(nn.Module):
	def __init__(self, input_dim, out_dim, num_layers):
		super(MutanFusion, self).__init__()
		self.input_dim = input_dim
		self.out_dim = out_dim
		self.num_layers = num_layers

		hv = []
		for i in range(self.num_layers):
			do = nn.Dropout(p=0.5)
			lin = nn.Linear(input_dim, out_dim)

			hv.append(nn.Sequential(do, lin, nn.Tanh()))

		self.image_transformation_layers = nn.ModuleList(hv)

		hq = []
		for i in range(self.num_layers):
			do = nn.Dropout(p=0.5)
			lin = nn.Linear(input_dim, out_dim)
			hq.append(nn.Sequential(do, lin, nn.Tanh()))

		self.text_transformation_layers = nn.ModuleList(hq)

	def forward(self, text_emb, img_emb):
		batch_size = img_emb.size()[0]
		x_mm = []
		for i in range(self.num_layers):
			x_hv = img_emb
			x_hv = self.image_transformation_layers[i](x_hv)

			x_hq = text_emb
			x_hq = self.text_transformation_layers[i](x_hq)
			x_mm.append(torch.mul(x_hq, x_hv))

		x_mm = torch.stack(x_mm, dim=1)
		x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
		x_mm = F.tanh(x_mm)
		return x_mm

class Normalize(nn.Module):
	def __init__(self, p=2):
		super(Normalize, self).__init__()
		self.p = p

	def forward(self, x):
		x = x / x.norm(p=self.p, dim=1, keepdim=True)
		return x

class ImageEmbedding(nn.Module):
	def __init__(self, image_channel_type='I', output_size=1024, extract_features=False, features_dir=None):
		super(ImageEmbedding, self).__init__()
		self.extractor = models.vgg16(pretrained=True)
		# freeze feature extractor (VGGNet) parameters
		for param in self.extractor.parameters():
			param.requires_grad = False

		extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
		if image_channel_type.lower() == 'normi':
			extactor_fc_layers.append(Normalize(p=2))
		self.extractor.classifier = nn.Sequential(*([nn.Flatten()] + extactor_fc_layers))
		self.fflayer = nn.Sequential(
			nn.Linear(4096, output_size),
			nn.Tanh())

		# TODO: Get rid of this hack
		self.extract_features = extract_features

	def forward(self, image):
		if self.extract_features:
			image = self.extractor(image)
		image_embedding = self.fflayer(image)
		return image_embedding

class TextEmbedding(nn.Module):
	def __init__(self, input_size=300, hidden_size=512, output_size=1024, num_layers=2, batch_first=True):
		super(TextEmbedding, self).__init__()
		# TODO: take as parameter
		self.bidirectional = True
		if num_layers == 1:
			self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
								batch_first=batch_first, bidirectional=self.bidirectional)

			if self.bidirectional:
				self.fflayer = nn.Sequential(
					nn.Linear(2 * num_layers * hidden_size, output_size),
					nn.Tanh())
		else:
			self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
								num_layers=num_layers, batch_first=batch_first)
			self.fflayer = nn.Sequential(
				nn.Linear(2 * num_layers * hidden_size, output_size),
				nn.Tanh())

	def forward(self, ques):
		_, hx = self.lstm(ques)
		lstm_embedding = torch.cat([hx[0], hx[1]], dim=2)
		text_embedding = lstm_embedding[0]
		if self.lstm.num_layers > 1 or self.bidirectional:
			for i in range(1, self.lstm.num_layers):
				text_embedding = torch.cat(
					[text_embedding, lstm_embedding[i]], dim=1)
			text_embedding = self.fflayer(text_embedding)
		return text_embedding

class VQAModel(nn.Module):

	def __init__(self, output_size, emb_size=1024, image_channel_type='I', text_channel_type='lstm', use_mutan=True, extract_img_features=True):
		super(VQAModel, self).__init__()

		self.word_emb_size = GLOVE_EMBEDDING_SIZE
		self.image_channel = ImageEmbedding(image_channel_type, output_size=emb_size, extract_features=extract_img_features)
		self.vocabulary = np.load(VOCABULARY_PATH)

		# NOTE the padding_idx below.
		self.word_embeddings = word_embedding_layer(self.vocabulary)
		if text_channel_type.lower() == 'lstm':
			self.text_channel = TextEmbedding(input_size=self.word_emb_size, output_size=emb_size, num_layers=1)
		elif text_channel_type.lower() == 'deeplstm':
			self.text_channel = TextEmbedding(input_size=self.word_emb_size, output_size=emb_size, num_layers=2)
		else:
			msg = 'text channel type not specified. please choose one of -  lstm or deeplstm'
			print(msg)
			raise Exception(msg)

		if use_mutan:
			self.mutan = MutanFusion(emb_size, emb_size, 5)
			self.mlp = nn.Sequential(nn.Linear(emb_size, output_size))
		else:
			self.mlp = nn.Sequential(
				nn.Linear(emb_size, 1000),
				nn.Dropout(p=0.5),
				nn.Tanh(),
				nn.Linear(1000, output_size))

	def forward(self, images, texts):
		image_embeddings = self.image_channel(images)
		embeds = self.word_embeddings(texts)
		text_embeddings = self.text_channel(embeds)
		if hasattr(self, 'mutan'):
			combined = self.mutan(text_embeddings, image_embeddings)
		else:
			combined = image_embeddings * text_embeddings
		output = self.mlp(combined)
		return output

	def saveCheckpoint(self, savePath, epoch, optimizer, bestTrainLoss, bestValLoss, isBest):
		ckpt = {}
		ckpt['state'] = self.state_dict()
		ckpt['epoch'] = epoch
		ckpt['optimizer_state'] = optimizer.state_dict()
		ckpt['bestTrainLoss'] = bestTrainLoss
		ckpt['bestValLoss'] = bestValLoss
		torch.save(ckpt, os.path.join(savePath, 'model.ckpt'))
		if isBest:
			torch.save(ckpt, os.path.join(savePath, 'bestModel.ckpt'))

	def loadCheckpoint(self, loadPath, optimizer, loadBest=False):
		if loadBest:
			ckpt = torch.load(os.path.join(loadPath, 'bestModel.ckpt'))
		else:
			ckpt = torch.load(os.path.join(loadPath, 'model.ckpt'))
		self.load_state_dict(ckpt['state'])
		epoch = ckpt['epoch']
		bestTrainLoss = ckpt['bestTrainLoss']
		bestValLoss = ckpt['bestValLoss']
		optimizer.load_state_dict(ckpt['optimizer_state'])
		return epoch, bestTrainLoss, bestValLoss