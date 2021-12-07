import os
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

import torchvision.transforms as T
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

import torch
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = '/dfs/user/paridhi/CS229/Project/Data'

csv_path_train = os.path.join(DATA_PATH, 'split_train.csv')
csv_path_val = os.path.join(DATA_PATH, 'split_val.csv')
csv_path_test = os.path.join(DATA_PATH, 'split_test.csv')

image_path = os.path.join(DATA_PATH, 'images_training')
IMAGE_DIMENSION = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

text_data_column = 'Text Normalized'
# text_data_column = 'Text Transcription'

target_names = ['non-misogynistic', 'misogynistic']

def read_data_from_csv(path_name):
	df = pd.read_csv(path_name, usecols=['file_name', 'misogynous', text_data_column], sep='\t')
	path = image_path+'/'
	df['image_path'] = path + df['file_name']
	return df

## Reading data
train_df = read_data_from_csv(csv_path_train)
val_df = read_data_from_csv(csv_path_val)
test_df = read_data_from_csv(csv_path_test)

print(train_df.shape)
extractor = models.vgg16(pretrained=True)

class ImageEmbedding(nn.Module):
	def __init__(self, image_channel_type='I', output_size=1024, extract_features=False, features_dir=None):
		super(ImageEmbedding, self).__init__()
		self.extractor = models.vgg16(pretrained=True).cuda()
		# freeze feature extractor (VGGNet) parameters
		for param in self.extractor.parameters():
			param.requires_grad = False

		extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
		if image_channel_type.lower() == 'normi':
			extactor_fc_layers.append(Normalize(p=2))
		self.extractor.classifier = nn.Sequential(*([nn.Flatten()] + extactor_fc_layers))
		self.fflayer = nn.Sequential(
			nn.Linear(4096, output_size),
			nn.Tanh()).cuda()

		# TODO: Get rid of this hack
		self.extract_features = extract_features

	def forward(self, image):
		if self.extract_features:
			image = self.extractor(image)
		image_embedding = self.fflayer(image)
		return image_embedding

emb_size = 1024
image_channel = ImageEmbedding(image_channel_type='I', output_size=emb_size, extract_features=True)

# image_path = train_df['image_path'][0]
transform = [T.Resize(IMAGE_DIMENSION), T.ToTensor()]
transform_fn = T.Compose(transform)



# extractor = models.vgg16(pretrained=True).to(device)



def convert_images_to_features(df):
    
    ## batch_size
    n = 64
    
    # using list comprehension 
    batch_data = [df['image_path'][i:i + n] for i in range(0, len(df['image_path']), n)] 
    # len(batch_data)
    
    features = []
    
    for batch_data_i in tqdm(batch_data):
        image_batch = []
        for image_path in batch_data_i:
            with open(image_path, 'rb') as f:
                with Image.open(f) as image:
                    image = image.convert('RGB')
                    image = transform_fn(image)
            #         
                    image = image.reshape(1,3,224,224)
                    image_batch.append(image)
        img_batch = torch.tensor(np.concatenate(image_batch, axis=0)).cuda()
        feature = image_channel(img_batch)
        features.append(feature.cpu().detach().numpy())
    
    features = np.concatenate(features, axis=0)
    return features

train_features = convert_images_to_features(train_df)
val_features = convert_images_to_features(val_df)
test_features = convert_images_to_features(test_df)

image_features = {
	'train_features': train_features,
	'val_features': val_features,
	'test_features': test_features
}

with open('image_features.pickle', 'wb') as handle:
    pickle.dump(image_features, handle, protocol=pickle.HIGHEST_PROTOCOL) 
