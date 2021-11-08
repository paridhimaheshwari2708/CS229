import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from config import VOCABULARY_PATH, GLOVE_PATH, URBAN_PATH, EMBEDDING_SIZE, OOV_SCALE

def word_embedding_layer(vocabulary, mode, trainable=True):
	words_in_dict = 0
	if mode == 'glove':
		embedding_map = np.load(GLOVE_PATH, allow_pickle=True)[()]
	elif mode == 'urban':
		embedding_map = np.load(URBAN_PATH, allow_pickle=True)[()]
	weight_matrix = np.zeros((len(vocabulary), EMBEDDING_SIZE))
	for i, word in enumerate(vocabulary):
		try:
			weight_matrix[i] = embedding_map[word]
			words_in_dict += 1
		except KeyError:
			# TODO: same random initialization for out-of-vocab words from test set
			weight_matrix[i] = np.random.normal(scale = OOV_SCALE, size = (EMBEDDING_SIZE, ))
	print('# of words: {}'.format(len(vocabulary)))
	print('# of words found: {}'.format(words_in_dict))

	embedding_layer = nn.Embedding(len(vocabulary), EMBEDDING_SIZE)
	embedding_layer.load_state_dict({'weight': torch.tensor(weight_matrix)})
	if not trainable:
		embedding_layer.weight.requires_grad = False
	return embedding_layer

class Normalize(nn.Module):
	def __init__(self, p=2):
		super(Normalize, self).__init__()
		self.p = p

	def forward(self, x):
		x = x / x.norm(p=self.p, dim=1, keepdim=True)
		return x

class ImageEmbedding(nn.Module):
	def __init__(self, image_channel_type='I', output_size=1024, extract_features=False, features_dir=None, mode='general'):
		super(ImageEmbedding, self).__init__()

		self.mode = mode

		if self.mode == 'general':
			self.extractor = models.vgg16(pretrained=True)
			# freeze feature extractor (VGGNet) parameters
			for param in self.extractor.parameters():
				param.requires_grad = False
			extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
			if image_channel_type.lower() == 'normi':
				extactor_fc_layers.append(Normalize(p=2))
			self.extractor.classifier = nn.Sequential(*([nn.Flatten()] + extactor_fc_layers))
			embedding_size = 4096
		elif self.mode == 'clip':
			self.extractor, _ = clip.load("ViT-B/32")
			# freeze feature extractor (VGGNet) parameters
			for param in self.extractor.parameters():
				param.requires_grad = False
			embedding_size = 512

		self.fflayer = nn.Sequential(
			nn.Linear(embedding_size, output_size),
			nn.Tanh())

		# TODO: Get rid of this hack
		self.extract_features = extract_features

	def forward(self, image):
		if self.extract_features
			if self.mode == 'general':
				image = self.extractor(image)
			elif self.mode == 'clip':
				image = self.extractor.encode_image(image).float()
		image_embedding = self.fflayer(image)
		return image_embedding

class ImageModel(nn.Module):

	def __init__(self, output_size, emb_size=1024, image_channel_type='I', extract_img_features=True, image_mode='general'):
		super(ImageModel, self).__init__()

		self.image_channel = ImageEmbedding(image_channel_type, output_size=emb_size, extract_features=extract_img_features, mode=image_mode)
		self.mlp = nn.Sequential(
			nn.Linear(emb_size, 512),
			nn.Dropout(p=0.5),
			nn.ReLU(),
			nn.Linear(512, output_size))

	def forward(self, images, texts):
		image_embeddings = self.image_channel(images)
		output = self.mlp(image_embeddings)
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

class TextModel(nn.Module):

	def __init__(self, output_size, emb_size=1024, text_channel_type='lstm', text_mode='glove'):
		super(TextModel, self).__init__()

		self.word_emb_size = EMBEDDING_SIZE
		self.vocabulary = np.load(VOCABULARY_PATH)

		# NOTE the padding_idx below.
		self.word_embeddings = word_embedding_layer(self.vocabulary, mode=text_mode)
		if text_channel_type.lower() == 'lstm':
			self.text_channel = TextEmbedding(input_size=self.word_emb_size, output_size=emb_size, num_layers=1)
		elif text_channel_type.lower() == 'deeplstm':
			self.text_channel = TextEmbedding(input_size=self.word_emb_size, output_size=emb_size, num_layers=2)
		else:
			msg = 'text channel type not specified. please choose one of -  lstm or deeplstm'
			print(msg)
			raise Exception(msg)

		self.mlp = nn.Sequential(
			nn.Linear(emb_size, 512),
			nn.Dropout(p=0.5),
			nn.ReLU(),
			nn.Linear(512, output_size))

	def forward(self, images, texts):
		embeds = self.word_embeddings(texts)
		text_embeddings = self.text_channel(embeds)
		output = self.mlp(text_embeddings)
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

class ImageTextModel(nn.Module):

	def __init__(self, output_size, emb_size=1024, image_channel_type='I', text_channel_type='lstm', use_mutan=True, extract_img_features=True, image_mode='general', text_mode='glove'):
		super(ImageTextModel, self).__init__()

		self.word_emb_size = EMBEDDING_SIZE
		self.image_channel = ImageEmbedding(image_channel_type, output_size=emb_size, extract_features=extract_img_features, mode=image_mode)
		self.vocabulary = np.load(VOCABULARY_PATH)

		# NOTE the padding_idx below.
		self.word_embeddings = word_embedding_layer(self.vocabulary, mode=text_mode)
		if text_channel_type.lower() == 'lstm':
			self.text_channel = TextEmbedding(input_size=self.word_emb_size, output_size=emb_size, num_layers=1)
		elif text_channel_type.lower() == 'deeplstm':
			self.text_channel = TextEmbedding(input_size=self.word_emb_size, output_size=emb_size, num_layers=2)
		else:
			msg = 'text channel type not specified. please choose one of -  lstm or deeplstm'
			print(msg)
			raise Exception(msg)

		self.mlp = nn.Sequential(
			nn.Linear(2*emb_size, 512),
			nn.Dropout(p=0.5),
			nn.ReLU(),
			nn.Linear(512, output_size))

	def forward(self, images, texts):
		image_embeddings = self.image_channel(images)
		embeds = self.word_embeddings(texts)
		text_embeddings = self.text_channel(embeds)
		combined = torch.cat((image_embeddings, text_embeddings), dim=1)
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