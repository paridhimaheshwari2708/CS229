import os
import sys
import ast
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

from config import DATA_PATH, IMAGE_DIR, IMAGE_DIMENSION, IMAGENET_MEAN, IMAGENET_STD

class Memes(Dataset):
	def __init__(self, subset, mode):
		super(Memes, self).__init__()

		self.mode = mode
		if self.mode == "TaskA":
			self.label_columns = ['misogynous']
		elif self.mode == 'TaskB':
			self.label_columns = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']

		csv_path = os.path.join(DATA_PATH, 'split_{}.csv'.format(subset))
		self.data = pd.read_csv(csv_path, sep='\t', \
			usecols=['file_name'] + self.label_columns + ['Text Transcription', 'Text Normalized', 'Text Indices'])

		print('{} set:\t# (image, text) pairs {}'.format(subset, len(self.data)))

		# transform = [T.Resize(IMAGE_DIMENSION), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
		transform = [T.Resize(IMAGE_DIMENSION), T.ToTensor()]
		self.transform = T.Compose(transform)

	def __getitem__(self, idx):
		curr = self.data.iloc[idx]
		image_path = curr['file_name']
		with open(os.path.join(IMAGE_DIR, image_path), 'rb') as f:
			with Image.open(f) as image:
				image = image.convert('RGB')
				image = self.transform(image)
		text_indices = curr['Text Indices']
		text = torch.tensor(ast.literal_eval(text_indices), dtype=torch.long)
		# BCEWithLogits Loss expects target in float
		labels = torch.tensor([curr[x] for x in self.label_columns], dtype=torch.float)
		return (image, text, labels)

	def __len__(self):
		return len(self.data)