import torch
import argparse
from torch.utils.data import DataLoader
from src.utils import *
from src import train
from src.metrics import Accuracy, Precision, Recall, F1Score

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"]="4,8"

# torch.multiprocessing.set_start_method('spawn')


parser = argparse.ArgumentParser(description='Meme Misoginy detection')

## Model
parser.add_argument('--model_mode', type=str, default='train',
                    help='Set if it is Train / Test')

parser.add_argument('--best_model_cpt', type=str, default='model',
                    help='Set if it is Train / Test')

# Fixed
parser.add_argument('--model', type=str, default='AverageBERT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--dataset', type=str, default='mami_dataset',
                    help='dataset to use (default: meme_dataset)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--mlp_dropout', type=float, default=0.0,
                    help='fully connected layers dropout')

# Architecture
parser.add_argument('--bert_model', type=str, default="bert-base-cased",
                    help='pretrained bert model to use')
parser.add_argument('--cnn_model', type=str, default="vgg16",
                    help='pretrained CNN to use for image feature extraction')
parser.add_argument('--image_feature_size', type=int, default=4096,
                    help='image feature size extracted from pretrained CNN (default: 4096), Use 512 for clip embeddings')
parser.add_argument('--bert_hidden_size', type=int, default=768,
                    help='bert hidden size for each word token (default: 768)')
parser.add_argument('--image_mode', type=str, default='general',
                    help='general / clip mode for the image embeddings')
parser.add_argument("--hierarchical", action="store", type=str, choices=["all", "true"], required=True)


# Tuning
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 8)')
parser.add_argument('--max_token_length', type=int, default=50,
                    help='max number of tokens per sentence (default: 50)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='initial learning rate (default: 2e-5)')
parser.add_argument('--optim', type=str, default='AdamW',
                    help='optimizer to use (default: AdamW)')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs (default: 3)')
parser.add_argument('--when', type=int, default=2,
                    help='when to decay learning rate (default: 2)')

# Logistics
parser.add_argument('--log_interval', type=int, default=100,
                    help='frequency of result logging (default: 100)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='model',
                    help='name of the trial (default: "model")')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers to use for DataLoaders (default: 6)')

args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
print(dataset)

use_cuda = False

if args.hierarchical == "all":
    num_classes = 5
elif args.hierarchical == "true":
    num_classes = 4

output_dim_dict = {
    'mami_dataset': 1,
    'mami_multi_dataset': num_classes, ## Changing for hierarchical dataset
    'meme_dataset': 2,
    'mmimdb': 27
}

criterion_dict = {
    'mami_dataset': 'BCEWithLogitsLoss',
    'mami_multi_dataset': 'BCEWithLogitsLoss',
    'meme_dataset': 'CrossEntropyLoss',
    'mmimdb': 'BCEWithLogitsLoss'

}

# torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset
#
####################################################################

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'val')
test_data = get_data(args, dataset, 'test')

train_loader = DataLoader(train_data,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)
valid_loader = DataLoader(valid_data,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)
if test_data is None:
    test_loader = None
else:
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

print('Finish loading the data....')


####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.n_train, hyp_params.n_valid = len(train_data), len(valid_data)
if test_data is None:
    pass
else:
    hyp_params.n_test = len(test_data)
hyp_params.model = args.model.strip()
hyp_params.output_dim = output_dim_dict.get(dataset)
hyp_params.criterion = criterion_dict.get(dataset)

print("Output dim is :", hyp_params.output_dim)
if __name__ == '__main__':
    if(args.model_mode == 'train'):
        test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
    if(args.model_mode == 'test'):
        train.test_model(hyp_params, test_loader)
