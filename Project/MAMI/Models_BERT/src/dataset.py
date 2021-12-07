import os
import torch
import clip
import json
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer

DATA_PATH = "/dfs/user/paridhi/CS229/Project/Data"
IMAGE_DIR = os.path.join(DATA_PATH, "images_training")

Image.MAX_IMAGE_PIXELS = 1000000000

class MAMIDataset(Dataset):
    """Hateful memes dataset from Facebook challenge"""

    def __init__(self, root_dir, dataset, split, model_name, max_len =512, mode= 'TaskA',image_mode = 'general', transform=None):
        """
        Args:
            jsonl_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.image_mode = image_mode
        # Metadata
        self.transform = transform

        # BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len

        self.mode = mode
        if self.mode == "TaskA":
            self.label_columns = ['misogynous']
        elif self.mode == 'TaskB':
            self.label_columns = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
        
        if self.image_mode == 'general':
            self.load_image = self.load_image_general
            # self.transform = transform
        elif self.image_mode == 'clip':
            self.load_image = self.load_image_clip
            _, self.preprocess = clip.load("ViT-B/32")

        csv_path = os.path.join(DATA_PATH, 'split_{}.csv'.format(split))
        self.data = pd.read_csv(csv_path, sep='\t', \
			usecols=['file_name'] + self.label_columns + ['Text Transcription', 'Text Normalized', 'Text Indices'])

    def load_image_general(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

    def load_image_clip(self, image_path):
        image = self.preprocess(Image.open(image_path))
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        curr = self.data.iloc[idx]
        img_name = curr['file_name']
        image_path = os.path.join(IMAGE_DIR, img_name)

        text = curr['Text Transcription']

        # BCEWithLogits Loss expects target in float
        labels = torch.tensor([curr[x] for x in self.label_columns], dtype=torch.float)
        image = self.load_image(image_path)
        sample = {'img_name':img_name,
                    'image': image,
                  'input_ids': text,
                  "label": labels}

        return sample

class MemeDataset(Dataset):
    """Hateful memes dataset from Facebook challenge"""

    def __init__(self, root_dir, dataset, split, model_name, max_len, transform=None):
        """
        Args:
            jsonl_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Metadata
        self.full_data_path = os.path.join(root_dir, dataset) + f'/{split}.jsonl'
        self.data_dict = pd.read_json(self.full_data_path, lines=True)
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform

        # BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir+'/'+self.dataset+'/'+self.data_dict.iloc[idx,1]
        image = Image.open(img_name).convert('RGB')
        label = self.data_dict.iloc[idx,2]

        text = self.data_dict.iloc[idx,3]
        text_encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.transform:
            image = self.transform(image)

        sample = {'image': image,
                  'input_ids': text_encoded['input_ids'].flatten(),
                  'attention_mask': text_encoded['attention_mask'].flatten(),
                  "label": int(label)}

        return sample
    
    
class MMIMDbDataset(Dataset):
    """Multimodal IMDb dataset (http://lisi1.unal.edu.co/mmimdb)"""

    def __init__(self, root_dir, dataset, split, transform=None):
        """
        Args:
            jsonl_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Metadata
        self.full_data_path = os.path.join(root_dir, dataset) + '/split.json'
        with open(self.full_data_path) as json_data:
            self.data_dict_raw = json.load(json_data)[split]
        
        plots = []
        image_names = []
        genres = []

        for id in self.data_dict_raw:
            with open(os.path.join(root_dir, dataset)+"/dataset/"+str(id)+'.json') as json_data:
                movie = json.load(json_data)
            plots.append(movie['plot'][0])
            genres.append(movie['genres'])
            image_names.append(os.path.join(root_dir, dataset)+"/dataset/"+str(id)+'.jpeg')
            
        self.data_dict = pd.DataFrame({'image': image_names, 'label': genres, 'text': plots})
            
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform
        self.genres = ['Horror', 'News', 'Animation',
                       'Musical', 'Fantasy', 'Family',
                       'Romance', 'Short', 'Comedy',
                       'Film-Noir', 'Mystery', 'Thriller',
                       'Documentary', 'Crime', 'History',
                       'Biography', 'Western', 'War',
                       'Adult', 'Adventure', 'Drama',
                       'Action', 'Music', 'Sci-Fi',
                       'Sport', 'Reality-TV', 'Talk-Show']
        self.num_classes = len(self.genres)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_dict.iloc[idx,0]
        image = Image.open(img_name).convert('RGB')
        
        label = self.data_dict.iloc[idx,1]
        indeces = torch.LongTensor([self.genres.index(e) for e in label])
        label = torch.nn.functional.one_hot(indeces, num_classes = self.num_classes).sum(dim=0)

        text = self.data_dict.iloc[idx,2]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image,
                  'input_ids': text,
                  "label": label.type(torch.FloatTensor)}

        return sample