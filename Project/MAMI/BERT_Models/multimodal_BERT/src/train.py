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
from sklearn.metrics import classification_report
from pprint import pprint
import csv
import pickle
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
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)

    settings = {'model': model,
                'bert': bert,
                'tokenizer': tokenizer,
                'feature_extractor': feature_extractor,
                'fflayer':fflayer,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train(hyp_params, train_loader, model, bert, tokenizer, feature_extractor,fflayer, optimizer, criterion, epoch):
    epoch_loss = 0
    model.train()
    num_batches = hyp_params.n_train // hyp_params.batch_size
    proc_loss, proc_size = 0, 0
    total_loss = 0.0
    losses = []
    results = []
    truths = []
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
                # print("FEATURE IMAGES", feature_images, feature_images.shape)
                # exit()

        if(hyp_params.image_mode == 'clip'):
            feature_images = feature_extractor.encode_image(images).float()
            # print("FEATURE IMAGES", feature_images, feature_images.shape)
            # feature_images = fflayer(feature_images)
            # print("AFTER FF LAYER",feature_images, feature_images.shape )
            # print(feature_images.shape)
            # exit()

        # last_hidden, pooled_output = bert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask
        # )
        bert_outputs = bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # print("HERE - pooled", bert_outputs.pooler_output, bert_outputs.pooler_output.shape)
        # print("LAST HIDDEN STATE - ", bert_outputs.last_hidden_state)

        outputs = model(
            last_hidden=bert_outputs.last_hidden_state,
            pooled_output=bert_outputs.pooler_output,
            feature_images=feature_images
        )

        sigmoid = nn.Sigmoid()
        if hyp_params.dataset == 'meme_dataset':
            _, preds = torch.max(outputs, dim=1)
        else:
            preds = outputs
            
        preds_round = (preds > 0.5).float()
        # print("OUTPUTS .. ",outputs, outputs.shape)
        # print("TARGETS .. ",targets, targets.shape)


        loss = criterion(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
        optimizer.step()
        #optimizer.zero_grad()
        
        total_loss += loss.item() * hyp_params.batch_size
        # results.append(preds)
        # truths.append(targets)
        results.append(sigmoid(preds).detach().cpu().numpy())
        truths.append(targets.detach().cpu().numpy())

        proc_loss += loss * hyp_params.batch_size
        proc_size += hyp_params.batch_size
        if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
            train_acc, train_f1 = METRICS['Accuracy'](preds_round, targets), METRICS['F1Score'](preds_round, targets)
            avg_loss = proc_loss / proc_size
            elapsed_time = time.time() - start_time
            print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | Train Acc {:5.4f} | Train f1-score {:5.4f}'.
                    format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss, train_acc, train_f1))
            proc_loss, proc_size = 0, 0
            start_time = time.time()
        
            
    avg_loss = total_loss / hyp_params.n_train
    # results = torch.cat(results)
    # truths = torch.cat(truths)
    results = np.concatenate(results, axis=0)
    truths = np.concatenate(truths, axis=0)

    return results, truths, avg_loss

def evaluate(hyp_params, data_loader, model, bert, tokenizer, feature_extractor,fflayer, criterion, test=False, details = False):
    model.eval()
    # loader = test_loader if test else valid_loader
    loader = data_loader
    total_loss = 0.0

    results = []
    truths = []
    correct_predictions = 0
    img_names_list = []
    with torch.no_grad():
        for i_batch, data_batch in enumerate(loader):
            input_ids = data_batch["input_ids"]
            targets = data_batch["label"]
            images = data_batch['image']
            img_name = data_batch['img_name']
            
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

            outputs = model(
            last_hidden=bert_outputs.last_hidden_state,
            pooled_output=bert_outputs.pooler_output,
            feature_images=feature_images
        )
            
            sigmoid = nn.Sigmoid()
            
            if hyp_params.dataset == 'meme_dataset':
                _, preds = torch.max(outputs, dim=1)
            else:
                preds = outputs
            
            total_loss += criterion(outputs, targets).item() * hyp_params.batch_size
            correct_predictions += torch.sum(preds == targets)

            # Collect the results into dictionary
            # results.append(preds)
            # truths.append(targets)

            results.append(sigmoid(preds).detach().cpu().numpy())
            truths.append(targets.detach().cpu().numpy())
            img_names_list.extend(img_name)

    avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

    # results = torch.cat(results)
    # truths = torch.cat(truths)

    results = np.concatenate(results, axis=0)
    truths = np.concatenate(truths, axis=0)

    if(details == False):
        return results, truths, avg_loss
    else:
        return results, truths, avg_loss, img_names_list

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    bert = settings['bert']
    tokenizer = settings['tokenizer']
    feature_extractor = settings['feature_extractor']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']

    fflayer = settings['fflayer']


    

    best_valid = 1e8
    writer = SummaryWriter('runs/'+hyp_params.model)
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_results, train_truths, train_loss = train(hyp_params, train_loader, model, bert, tokenizer, feature_extractor,fflayer, optimizer, criterion, epoch)
        results, truths, val_loss = evaluate(hyp_params, valid_loader, model, bert, tokenizer, feature_extractor,fflayer, criterion, test=False)
        #if test_loader is not None:
        #    results, truths, val_loss = evaluate(model, feature_extractor, criterion, test=True)

        end = time.time()
        duration = end-start
        scheduler.step(val_loss)

        # train_acc, train_f1 = metrics(train_results, train_truths) 
        # val_acc, val_f1 = metrics(results, truths)

        train_acc, train_f1 = METRICS['Accuracy'](train_results, train_truths), METRICS['F1Score'](train_results, train_truths)
        val_acc, val_f1 = METRICS['Accuracy'](results, truths), METRICS['F1Score'](results, truths)

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Valid Acc {:5.4f} | Valid f1-score {:5.4f}'.format(epoch, duration, train_loss, val_loss, val_acc, val_f1))
        print("-"*50)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('F1-score/train', train_f1, epoch)

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1-score/val', val_f1, epoch)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss
    
    print(f"Saved the final model at pre_trained_models/{hyp_params.name}_final.pt!")
    save_model(hyp_params, model, name=hyp_params.name+'_final')

    if test_loader is not None:
        model = load_model(hyp_params, name=hyp_params.name)
        results, truths, val_loss = evaluate(hyp_params, test_loader, model, bert, tokenizer, feature_extractor,fflayer, criterion, test=True)
        # test_acc, test_f1 = metrics(results, truths)
        test_acc, test_f1 = METRICS['Accuracy'](results, truths), METRICS['F1Score'](results, truths)
        
        print("\n\nTest Acc {:5.4f} | Test f1-score {:5.4f}".format(test_acc, test_f1))

    sys.stdout.flush()

def test_model(hyp_params, test_loader):

    model = getattr(models, hyp_params.model+'Model')(hyp_params)
    bert = BertModel.from_pretrained(hyp_params.bert_model)
    tokenizer = BertTokenizer.from_pretrained(hyp_params.bert_model)
    criterion = getattr(nn, hyp_params.criterion)()

    # feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', hyp_params.cnn_model, pretrained=True)

    if(hyp_params.image_mode == 'general'):
        feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', hyp_params.cnn_model, pretrained=True)

    if(hyp_params.image_mode == 'clip'):
        feature_extractor, _ = clip.load("ViT-B/32")

    model = load_model(hyp_params, name=hyp_params.best_model_cpt)
    results, truths, test_loss, img_names = evaluate(hyp_params, test_loader, model, bert, tokenizer, feature_extractor,fflayer =None, criterion = criterion, test=True, details=True)
    # test_acc, test_f1 = metrics(results, truths)
    metric_results = {"Loss": test_loss}
    for metric, metric_fn in METRICS.items():
        metric_results[metric] = metric_fn(results, truths)
    
    pprint(metric_results)

    if hyp_params.dataset == "mami_dataset":
        target_names = ['not_misogynous','misogynous']
    elif hyp_params.dataset == 'mami_multi_dataset':
        target_names = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
    
    predictions = (results > 0.5).astype(float)
    print(classification_report(truths, predictions, target_names=target_names))
    
    write_data = []
    print(len(img_names))
    for name, pred, gt in zip(img_names, predictions, truths):
        print(name, pred.tolist(), gt.tolist())
        pred_list = [int(x) for x in pred.tolist()]
        gt_list = [int(x) for x in gt.tolist()]
        # print(type(pred))
        # print(pred.tolist())
        write_data_row = [name, pred_list, gt_list]
        write_data.append(write_data_row)
    
    with open('./gated_avg_clip_result.pkl','wb') as f:
        pickle.dump(write_data, f)

    # import pdb
    # pdb.set_trace()

    header = ['image_name', 'predictions', 'ground_truth']
    
    with open('gated_avg_clip_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(write_data)
    # test_acc, test_f1 = METRICS['Accuracy'](results, truths), METRICS['F1Score'](results, truths)

    # print("\n\nTest Acc {:5.4f} | Test f1-score {:5.4f}".format(test_acc, test_f1))
    return