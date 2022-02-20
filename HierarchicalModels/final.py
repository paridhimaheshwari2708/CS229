import os
import ast
import csv
import clip
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from vqa import VQAModel
from san import SANModel
from unimodal import TextModel, ImageModel, ImageTextModel
from config import DATA_PATH, IMAGE_DIMENSION, IMAGENET_MEAN, IMAGENET_STD

IMAGE_DIR = os.path.join(DATA_PATH, "images_final")

torch.autograd.set_detect_anomaly(True)

class Options:
	def __init__(self):
		self.parser = argparse.ArgumentParser(description="Training Autoencoder")
		self.parser.add_argument("--load", dest="load", action="store", required=True)
		self.parser.add_argument("--image_mode", action="store", type=str, choices=["general", "clip"], required=True)
		self.parser.add_argument("--text_mode", action="store", type=str, choices=["glove", "urban"], required=True)
		self.parser.add_argument("--model", action="store", type=str, choices=["VQA", "MUTAN", "SAN", "Text", "Image", "ImageText"], required=True)
		self.parser.add_argument("--hierarchical", action="store", type=str, choices=["all", "true"], required=True)
		self.parser.add_argument("--batchSize", dest="batchSize", action="store", default=64, type=int)
		self.parser.add_argument("--numWorkers", dest="numWorkers", action="store", default=16, type=int)
		self.parser.add_argument("--lr", dest="lr", action="store", default=0.001, type=float)

		self.parse()
		self.checkArgs()

	def parse(self):
		self.opts = self.parser.parse_args()

	def checkArgs(self):
		if self.opts.load:
			assert os.path.exists(os.path.join("logs", self.opts.load)), "Load Path doesn't Exist"

	def __str__(self):
		return ("All Options:\n"+ "".join(["-"] * 45)+ "\n"+ "\n".join(["{:<18} -------> {}".format(k, v) for k, v in vars(self.opts).items()])+ "\n"+ "".join(["-"] * 45)+ "\n")


class Memes(Dataset):
	def __init__(self, subset, image_mode):
		super(Memes, self).__init__()

		self.image_mode = image_mode

		csv_path = os.path.join(DATA_PATH, 'split_{}.csv'.format(subset))
		self.data = pd.read_csv(csv_path, sep='\t', usecols=['file_name', 'Text Transcription', 'Text Normalized', 'Text Indices'])

		print('{} set:\t# (image, text) pairs {}'.format(subset, len(self.data)))

		if self.image_mode == 'general':
			self.load_image = self.load_image_general
			# transform = [T.Resize(IMAGE_DIMENSION), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
			transform = [T.Resize(IMAGE_DIMENSION), T.ToTensor()]
			self.transform = T.Compose(transform)
		elif self.image_mode == 'clip':
			self.load_image = self.load_image_clip
			_, self.preprocess = clip.load("ViT-B/32")

	def load_image_general(self, image_path):
		with open(os.path.join(IMAGE_DIR, image_path), 'rb') as f:
			with Image.open(f) as image:
				image = image.convert('RGB')
				image = self.transform(image)
		return image

	def load_image_clip(self, image_path):
		image = self.preprocess(Image.open(os.path.join(IMAGE_DIR, image_path)))
		return image

	def __getitem__(self, idx):
		curr = self.data.iloc[idx]
		image_path = curr['file_name']
		image = self.load_image(image_path)
		text_indices = curr['Text Indices']
		text = torch.tensor(ast.literal_eval(text_indices), dtype=torch.long)
		return (image, text)

	def __len__(self):
		return len(self.data)


def buildModel(args, loadBest):
	if args.hierarchical == "all":
		num_classes = 5
	elif args.hierarchical == "true":
		num_classes = 4

	if args.model == "VQA":
		model = VQAModel(output_size=num_classes, use_mutan=False, image_mode=args.image_mode, text_mode=args.text_mode).cuda()
	elif args.model == "MUTAN":
		model = VQAModel(output_size=num_classes, use_mutan=True, image_mode=args.image_mode, text_mode=args.text_mode).cuda()
	elif args.model == "SAN":
		model = SANModel(output_size=num_classes, text_mode=args.text_mode).cuda()
	elif args.model == "Text":
		model = TextModel(output_size=num_classes, text_mode=args.text_mode).cuda()
	elif args.model == "Image":
		model = ImageModel(output_size=num_classes, image_mode=args.image_mode).cuda()
	elif args.model == "ImageText":
		model = ImageTextModel(output_size=num_classes, image_mode=args.image_mode, text_mode=args.text_mode).cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	if args.load:
		epoch, bestTrainLoss, bestValLoss = model.loadCheckpoint(os.path.join("logs", args.load), optimizer, loadBest)
	else:
		epoch, bestTrainLoss, bestValLoss = -1, float("inf"), float("inf")
	return epoch, bestTrainLoss, bestValLoss, optimizer, model


def predict(args, dataloader, model):
	model.eval()

	predictions_a, predictions_b = [], []

	sigmoid = nn.Sigmoid()
	for (image, text) in tqdm(dataloader):
		image, text = image.cuda(), text.cuda()
		preds_a, preds_b = model(image, text)

		if args.hierarchical == "true":
			non_select_idx = (preds_a < 0).squeeze()
			preds_b[non_select_idx, :] = -float("Inf")
			preds_b = torch.cat((preds_a, preds_b), dim=1)

		# Keep track of things
		predictions_a.append(sigmoid(preds_a).detach().cpu().numpy())
		predictions_b.append(sigmoid(preds_b).detach().cpu().numpy())

	# Gather the results
	predictions_a = np.concatenate(predictions_a, axis=0)
	predictions_b = np.concatenate(predictions_b, axis=0)

	# Threshold to get binary output
	predictions_a = (predictions_a > 0.5).astype(int)
	predictions_b = (predictions_b > 0.5).astype(int)
	return predictions_a, predictions_b

def test(args):
	_, _, _, _, model = buildModel(args, loadBest=True)

	testLoader = DataLoader(Memes("final", args.image_mode), 
							shuffle=False, num_workers=args.numWorkers, batch_size=args.batchSize)

	predictions_a, predictions_b = predict(args, testLoader, model)

	filenames = np.array(pd.read_csv(os.path.join(DATA_PATH, 'split_final.csv'), sep='\t', usecols=['file_name']))

	predictions_a = np.concatenate((filenames, predictions_a), axis=1)
	with open(os.path.join("logs", args.load, "final_a.txt"), 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerows(predictions_a)

	predictions_b = np.concatenate((filenames, predictions_b), axis=1)
	with open(os.path.join("logs", args.load, "final_b.txt"), 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerows(predictions_b)


if __name__ == "__main__":

	args = Options()
	print(args)

	test(args.opts)
