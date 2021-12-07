from src import models
from src.utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import torch
from transformers import BertModel
from transformers import BertTokenizer
import numpy as np
import time
import sys
import clip
from src.metrics import Accuracy, Precision, Recall, F1Score

torch.autograd.set_detect_anomaly(True)

METRICS = {
	"Accuracy": Accuracy(),
	"Precision": Precision(),
	"Recall": Recall(),
	"F1Score": F1Score(),
}

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *

import os

ALPHA = 1
CLASS_POS_WEIGHTS = [1.0, 6.849293563579278, 2.5587188612099645, 3.541326067211626, 9.49317943336831]

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,2"

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader=None):
	model = getattr(models, hyp_params.model+'Model')(hyp_params)
	bert = BertModel.from_pretrained(hyp_params.bert_model)
	tokenizer = BertTokenizer.from_pretrained(hyp_params.bert_model)

	fflayer = nn.Sequential(nn.Linear(512, 1024),nn.Tanh())

	if(hyp_params.image_mode == 'general'):
		feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', hyp_params.cnn_model, pretrained=True)
		for param in feature_extractor.features.parameters():
			param.requires_grad = False

	if(hyp_params.image_mode == 'clip'):
		feature_extractor, _ = clip.load("ViT-B/32")
		for param in feature_extractor.parameters():
			param.requires_grad = False

	if hyp_params.use_cuda:
		model = model.cuda()
		bert = bert.cuda()
		feature_extractor = feature_extractor.cuda()
		fflayer = fflayer.cuda()

	optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
	# criterion = getattr(nn, hyp_params.criterion)()
	criterion1 = nn.BCEWithLogitsLoss()
	if hyp_params.hierarchical == "all":
		criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CLASS_POS_WEIGHTS).cuda())
	elif hyp_params.hierarchical == "true":
		criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CLASS_POS_WEIGHTS[1:]).cuda())

	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)

	settings = {'model': model,
				'bert': bert,
				'tokenizer': tokenizer,
				'feature_extractor': feature_extractor,
				'fflayer':fflayer,
				'optimizer': optimizer,
				'criterion1': criterion1,
				'criterion2': criterion2,
				'scheduler': scheduler}

	return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train(hyp_params, train_loader, model, bert, tokenizer, feature_extractor,fflayer, optimizer, criterion1,criterion2, epoch):
	epoch_loss = 0
	model.train()
	num_batches = hyp_params.n_train // hyp_params.batch_size
	proc_loss, proc_size = 0, 0
	total_loss = 0.0
	losses = []
	results_a, results_b = [],[]
	truths_a, truths_b = [], []
	n_examples = hyp_params.n_train
	start_time = time.time()

	for i_batch, data_batch in enumerate(train_loader):
		
		input_ids = data_batch["input_ids"]
		targets = data_batch["label"]
		images = data_batch['image']
		
		text_encoded = tokenizer.batch_encode_plus(
			input_ids,
			truncation=True,
			add_special_tokens=True,
			max_length=hyp_params.max_token_length,
			return_token_type_ids=False,
			# pad_to_max_length=True,
			padding = 'max_length',
			return_attention_mask=True,
			return_tensors='pt',
		)

		model.zero_grad()

		if hyp_params.use_cuda:
			with torch.cuda.device(0):
				input_ids = text_encoded['input_ids'].cuda()
				attention_mask = text_encoded['attention_mask'].cuda()
				targets = targets.cuda()
				images = images.cuda()

		if images.size()[0] != input_ids.size()[0]:
			continue

		if(hyp_params.image_mode == 'general'):
			with torch.no_grad():
				feature_images = feature_extractor.features(images)
				feature_images = feature_extractor.avgpool(feature_images)
				feature_images = torch.flatten(feature_images, 1)
				feature_images = feature_extractor.classifier[0](feature_images)

		if(hyp_params.image_mode == 'clip'):
			feature_images = feature_extractor.encode_image(images).float()

		bert_outputs = bert(
			input_ids=input_ids,
			attention_mask=attention_mask
		)

		preds1, preds2  = model(
			last_hidden=bert_outputs.last_hidden_state,
			pooled_output=bert_outputs.pooler_output,
			feature_images=feature_images
		)

		sigmoid = nn.Sigmoid()

		
		loss1 = criterion1(preds1,targets[:,0].unsqueeze(1) )

		# print(preds1.shape, preds2.shape, targets.shape)
		if hyp_params.hierarchical == "all":
			loss2 = criterion2(preds2, targets)
			preds_a = preds1
			preds_b = preds2
		elif hyp_params.hierarchical == "true":
			select_idx = (targets[:,0] == 1)
			loss2 = criterion2(preds2[select_idx], targets[select_idx, 1:])
			non_select_idx = (preds1 < 0).squeeze()
			preds2[non_select_idx, :] = -float("Inf")
			preds_a = preds1
			preds_b = torch.cat((preds1, preds2), dim=1)
		
		loss = loss1 + ALPHA * loss2


		losses.append(loss.item())
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
		optimizer.step()
		#optimizer.zero_grad()

		# import pdb
		# pdb.set_trace()
		
		total_loss += loss.item() * hyp_params.batch_size
		# results.append(preds)
		# truths.append(targets)
		preds_a_round = sigmoid(preds_a).detach().cpu().numpy()
		preds_b_round = sigmoid(preds_b).detach().cpu().numpy()
		targets_a = targets[:,0].unsqueeze(1).detach().cpu().numpy()

		results_a.append(preds_a_round)
		results_b.append(preds_b_round)
		truths_a.append(targets_a)
		truths_b.append(targets.detach().cpu().numpy())

		proc_loss += loss * hyp_params.batch_size
		proc_size += hyp_params.batch_size


		# import pdb
		# pdb.set_trace()
		
		if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
			train_acc_a, train_f1_a = METRICS['Accuracy'](preds_a_round, targets_a), METRICS['F1Score'](preds_a_round, targets_a)
			train_acc_b, train_f1_b = METRICS['Accuracy'](preds_b_round, targets), METRICS['F1Score'](preds_b_round, targets)
			avg_loss = proc_loss / proc_size
			elapsed_time = time.time() - start_time
			print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | Train Acc_A {:5.4f} | Train f1-score _A {:5.4f}| Train Acc_B {:5.4f} | Train f1-score _B {:5.4f}'.
					format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss, train_acc_a, train_f1_a,train_acc_b, train_f1_b))
			proc_loss, proc_size = 0, 0
			start_time = time.time()

		
		
	avg_loss = total_loss / hyp_params.n_train
	# results = torch.cat(results)
	# truths = torch.cat(truths)
	results_a = np.concatenate(results_a, axis=0)
	results_b= np.concatenate(results_b, axis=0)
	truths_a = np.concatenate(truths_a, axis=0)
	truths_b = np.concatenate(truths_b, axis=0)

	return results_a,results_b, truths_a,truths_b, avg_loss

def evaluate(hyp_params, data_loader, model, bert, tokenizer, feature_extractor,fflayer, criterion1,criterion2, test=False):
	model.eval()
	# loader = test_loader if test else valid_loader
	loader = data_loader
	total_loss = 0.0

	results_a, results_b = [],[]
	truths_a, truths_b = [], []
	correct_predictions = 0

	with torch.no_grad():
		for i_batch, data_batch in enumerate(loader):
			input_ids = data_batch["input_ids"]
			targets = data_batch["label"]
			images = data_batch['image']
			
			text_encoded = tokenizer.batch_encode_plus(
				input_ids,
				add_special_tokens=True,
				truncation=True,
				max_length=hyp_params.max_token_length,
				return_token_type_ids=False,
				# pad_to_max_length=True,
				padding = 'max_length',
				return_attention_mask=True,
				return_tensors='pt',
			)

			if hyp_params.use_cuda:
				with torch.cuda.device(0):
					input_ids = text_encoded['input_ids'].cuda()
					attention_mask = text_encoded['attention_mask'].cuda()
					targets = targets.cuda()
					images = images.cuda()

			if images.size()[0] != input_ids.size()[0]:
				continue

			if(hyp_params.image_mode == 'general'):
				with torch.no_grad():
					feature_images = feature_extractor.features(images)
					feature_images = feature_extractor.avgpool(feature_images)
					feature_images = torch.flatten(feature_images, 1)
					feature_images = feature_extractor.classifier[0](feature_images)

			if(hyp_params.image_mode == 'clip'):
				with torch.no_grad():
					feature_images = feature_extractor.encode_image(images).float()
				
			bert_outputs = bert(
				input_ids=input_ids,
				attention_mask=attention_mask
			)

			# outputs = model(
			#     last_hidden=last_hidden,
			#     pooled_output=pooled_output,
			#     feature_images=feature_images
			# )

			preds1, preds2  = model(
			last_hidden=bert_outputs.last_hidden_state,
			pooled_output=bert_outputs.pooler_output,
			feature_images=feature_images
		)
			
			sigmoid = nn.Sigmoid()

			targets_a = targets[:,0].unsqueeze(1)
			loss1 = criterion1(preds1,targets_a )

			if hyp_params.hierarchical == "all":
				loss2 = criterion2(preds2, targets)
				preds_a = preds1
				preds_b = preds2
			elif hyp_params.hierarchical == "true":
				select_idx = (targets[:,0] == 1)
				loss2 = criterion2(preds2[select_idx], targets[select_idx, 1:])
				non_select_idx = (preds1 < 0).squeeze()
				preds2[non_select_idx, :] = -float("Inf")
				preds_a = preds1
				preds_b = torch.cat((preds1, preds2), dim=1)
			
			loss = loss1 + ALPHA * loss2
			
			total_loss += loss * hyp_params.batch_size
			# correct_predictions += torch.sum(preds == targets)

			results_a.append(sigmoid(preds_a).detach().cpu().numpy())
			results_b.append(sigmoid(preds_b).detach().cpu().numpy())
			truths_a.append(targets_a.detach().cpu().numpy())
			truths_b.append(targets.detach().cpu().numpy())

	avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)


	results_a = np.concatenate(results_a, axis=0)
	results_b= np.concatenate(results_b, axis=0)
	truths_a = np.concatenate(truths_a, axis=0)
	truths_b = np.concatenate(truths_b, axis=0)

	return results_a, results_b, truths_a, truths_b, avg_loss

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
	model = settings['model']
	bert = settings['bert']
	tokenizer = settings['tokenizer']
	feature_extractor = settings['feature_extractor']
	optimizer = settings['optimizer']
	# criterion = settings['criterion']
	criterion1 = settings['criterion1']
	criterion2 = settings['criterion2']

	scheduler = settings['scheduler']

	fflayer = settings['fflayer']

	best_valid = 1e8
	writer = SummaryWriter('runs/'+hyp_params.model)
	for epoch in range(1, hyp_params.num_epochs+1):
		start = time.time()
		train_results_a,train_results_b, train_truths_a, train_truths_b, train_loss = train(hyp_params, train_loader, model, bert, tokenizer, feature_extractor,fflayer, optimizer, criterion1,criterion2 , epoch)
		results_a,results_b , truths_a, truths_b, val_loss = evaluate(hyp_params, valid_loader, model, bert, tokenizer, feature_extractor,fflayer, criterion1,criterion2, test=False)

		end = time.time()
		duration = end-start
		scheduler.step(val_loss)

		# train_acc, train_f1 = metrics(train_results, train_truths) 
		# val_acc, val_f1 = metrics(results, truths)

		train_acc_a, train_f1_a = METRICS['Accuracy'](train_results_a, train_truths_a), METRICS['F1Score'](train_results_a, train_truths_a)
		val_acc_a, val_f1_a = METRICS['Accuracy'](results_a, truths_a), METRICS['F1Score'](results_a, truths_a)

		train_acc_b, train_f1_b = METRICS['Accuracy'](train_results_b, train_truths_b), METRICS['F1Score'](train_results_b, train_truths_b)
		val_acc_b, val_f1_b = METRICS['Accuracy'](results_b, truths_b), METRICS['F1Score'](results_b, truths_b)
		
		# train_acc, train_f1 = METRICS['Accuracy'](train_results, train_truths), METRICS['F1Score'](train_results, train_truths)
		# val_acc, val_f1 = METRICS['Accuracy'](results, truths), METRICS['F1Score'](results, truths)

		print("-"*50)
		print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Valid Acc Task B {:5.4f} | Valid f1-score Task B {:5.4f}'.format(epoch, duration, train_loss, val_loss, val_acc_b, val_f1_b))
		print('Valid Acc Task A {:5.4f} | Valid f1-score Task A {:5.4f}'.format(val_acc_a, val_f1_a))

		print("-"*50)
		
		writer.add_scalar('Loss/train', train_loss, epoch)
		writer.add_scalar('Accuracy_A/train', train_acc_a, epoch)
		writer.add_scalar('F1-score_A/train', train_f1_a, epoch)
		writer.add_scalar('Accuracy_B/train', train_acc_b, epoch)
		writer.add_scalar('F1-score_B/train', train_f1_b, epoch)

		writer.add_scalar('Loss/val', val_loss, epoch)
		writer.add_scalar('Accuracy_A/val', val_acc_a, epoch)
		writer.add_scalar('F1-score_A/val', val_f1_a, epoch)
		writer.add_scalar('Accuracy_B/val', val_acc_b, epoch)
		writer.add_scalar('F1-score_B/val', val_f1_b, epoch)

		if val_loss < best_valid:
			print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
			save_model(hyp_params, model, name=hyp_params.name)
			best_valid = val_loss
	
	print(f"Saved the final model at pre_trained_models/{hyp_params.name}_final.pt!")
	save_model(hyp_params, model, name=hyp_params.name+'_final')

	if test_loader is not None:
		model = load_model(hyp_params, name=hyp_params.name)
		results_a, results_b, truths_a, truths_b, val_loss = evaluate(hyp_params, test_loader, model, bert, tokenizer, feature_extractor,fflayer, criterion1,criterion2, test=True)
		# test_acc, test_f1 = metrics(results, truths)
		test_acc_b, test_f1_b = METRICS['Accuracy'](results_b, truths_b), METRICS['F1Score'](results_b, truths_b)
		test_acc_a, test_f1_a = METRICS['Accuracy'](results_a, truths_a), METRICS['F1Score'](results_b, truths_a)
		
		print("\n\nTest Acc Task A {:5.4f} | Test f1-score Task A{:5.4f}| Test Acc Task B {:5.4f} | Test f1-score Task B{:5.4f}".format(test_acc_a, test_f1_a,test_acc_b, test_f1_b))

	sys.stdout.flush()

def test_model(hyp_params, test_loader):

	model = getattr(models, hyp_params.model+'Model')(hyp_params)
	bert = BertModel.from_pretrained(hyp_params.bert_model)
	tokenizer = BertTokenizer.from_pretrained(hyp_params.bert_model)
	# criterion = getattr(nn, hyp_params.criterion)()
	criterion1 = nn.BCEWithLogitsLoss()
	if hyp_params.hierarchical == "all":
		criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CLASS_POS_WEIGHTS).cuda())
	elif hyp_params.hierarchical == "true":
		criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CLASS_POS_WEIGHTS[1:]).cuda())

	# feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', hyp_params.cnn_model, pretrained=True)

	if(hyp_params.image_mode == 'general'):
		feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', hyp_params.cnn_model, pretrained=True)

	if(hyp_params.image_mode == 'clip'):
		feature_extractor, _ = clip.load("ViT-B/32")

	model = load_model(hyp_params, name=hyp_params.best_model_cpt)
	results_a, results_b, truths_a, truths_b, test_loss = evaluate(hyp_params, test_loader, model, bert, tokenizer, feature_extractor,fflayer =None, criterion1 = criterion1, criterion2 = criterion2, test=True)
	# test_acc, test_f1 = metrics(results, truths)
	metric_results = {"Loss": test_loss}
	print(results_a.shape, results_b.shape, truths_a.shape, truths_b.shape)
	for metric, metric_fn in METRICS.items():
		metric_results[metric+'_a'] = metric_fn(results_a, truths_a)
		metric_results[metric+'_b'] = metric_fn(results_b, truths_b)
	print(metric_results)
	return
