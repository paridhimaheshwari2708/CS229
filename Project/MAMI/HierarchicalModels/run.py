'''
CUDA_VISIBLE_DEVICES=0 python run.py --save ImageText_all --model ImageText --image_mode general --text_mode glove --hierarchical all
CUDA_VISIBLE_DEVICES=0 python run.py --save VQA_all --model VQA --image_mode general --text_mode glove --hierarchical all
CUDA_VISIBLE_DEVICES=0 python run.py --save MUTAN_all --model MUTAN --image_mode general --text_mode glove --hierarchical all

CUDA_VISIBLE_DEVICES=0 python run.py --save ImageText_cwk_all --model ImageText --image_mode clip --text_mode urban --hierarchical all
CUDA_VISIBLE_DEVICES=0 python run.py --save VQA_cwk_all --model VQA --image_mode clip --text_mode urban --hierarchical all
CUDA_VISIBLE_DEVICES=0 python run.py --save MUTAN_cwk_all --model MUTAN --image_mode clip --text_mode urban --hierarchical all
'''

import os
import json
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import trange
from pprint import pprint
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from config import *
from data import Memes
from vqa import VQAModel
from san import SANModel
from unimodal import TextModel, ImageModel, ImageTextModel
from metrics import Accuracy, Precision, Recall, F1Score

torch.autograd.set_detect_anomaly(True)

METRICS = {
    "Accuracy": Accuracy(),
    "Precision": Precision(),
    "Recall": Recall(),
    "F1Score": F1Score(),
}

def set_seed(seed):
    """Utility function to set seed values for RNG for various modules"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training Autoencoder")
        self.parser.add_argument("--save", dest="save", action="store", required=True)
        self.parser.add_argument("--load", dest="load", action="store")
        self.parser.add_argument("--image_mode", action="store", type=str, choices=["general", "clip"], required=True)
        self.parser.add_argument("--text_mode", action="store", type=str, choices=["glove", "urban"], required=True)
        self.parser.add_argument("--model", action="store", type=str, choices=["VQA", "MUTAN", "SAN", "Text", "Image", "ImageText"], required=True)
        self.parser.add_argument("--hierarchical", action="store", type=str, choices=["all", "true"], required=True)
        self.parser.add_argument("--lr", dest="lr", action="store", default=0.001, type=float)
        self.parser.add_argument("--epochs", dest="epochs", action="store", default=20, type=int)
        self.parser.add_argument("--batchSize", dest="batchSize", action="store", default=64, type=int)
        self.parser.add_argument("--numWorkers", dest="numWorkers", action="store", default=16, type=int)

        self.parse()
        self.checkArgs()

    def parse(self):
        self.opts = self.parser.parse_args()

    def checkArgs(self):
        # Check Load and Save Paths
        if self.opts.load:
            assert os.path.exists(os.path.join("logs", self.opts.load)), "Load Path doesn't Exist"
        if self.opts.save:
            if not os.path.isdir(os.path.join("runs", self.opts.save)):
                os.makedirs(os.path.join("runs", self.opts.save))
            if not os.path.isdir(os.path.join("logs", self.opts.save)):
                os.makedirs(os.path.join("logs", self.opts.save))

    def __str__(self):
        return ("All Options:\n"+ "".join(["-"] * 45)+ "\n"+ "\n".join(["{:<18} -------> {}".format(k, v) for k, v in vars(self.opts).items()])+ "\n"+ "".join(["-"] * 45)+ "\n")


def buildLoader(args, subset):
    shuffle = (subset != "test")
    loader = DataLoader(
        Memes(subset, args.image_mode),
        shuffle=shuffle,
        num_workers=args.numWorkers,
        batch_size=args.batchSize,
    )
    return loader


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

def run(args, epoch, mode, dataloader, model, optimizer):
    if mode == "train":
        model.train()
    elif mode == "val" or mode == "test":
        model.eval()
    else:
        assert False, "Wrong Mode:{} for Run".format(mode)

    losses, predictions, targets = [], [], []
    criterion1 = nn.BCEWithLogitsLoss()
    if args.hierarchical == "all":
        criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CLASS_POS_WEIGHTS).cuda())
    elif args.hierarchical == "true":
        criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CLASS_POS_WEIGHTS[1:]).cuda())

    sigmoid = nn.Sigmoid()
    with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
        for (image, text, labels) in dataloader:
            image, text, labels = image.cuda(), text.cuda(), labels.cuda()
            preds1, preds2 = model(image, text)

            loss1 = criterion1(preds1, labels[:,0].unsqueeze(1))
            if args.hierarchical == "all":
                loss2 = criterion2(preds2, labels)
                preds = preds2
            elif args.hierarchical == "true":
                select_idx = (labels[:,0] == 1)
                loss2 = criterion2(preds2[select_idx], labels[select_idx, 1:])
                non_select_idx = (preds1 < 0).squeeze()
                preds2[non_select_idx, :] = -float("Inf")
                preds = torch.cat((preds1, preds2), dim=1)
            loss = loss1 + ALPHA * loss2

            if mode == "train":
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Keep track of things
            predictions.append(sigmoid(preds).detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())
            losses.append(loss.item())
            t.set_postfix(loss=losses[-1])
            t.update()
    # Gather the results for the epoch
    epoch_loss = sum(losses) / len(losses)
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    save_path = os.path.join("logs", args.save, "outputs.npz")
    np.savez(save_path, predictions=predictions, targets=targets)
    results = {"Loss": epoch_loss}
    for metric, metric_fn in METRICS.items():
        results[metric+"_a"] = metric_fn(predictions[:,0], targets[:,0])
        results[metric+"_b"] = metric_fn(predictions[:,1:], targets[:,1:])
    return results


def train(args):
    startingEpoch, bestTrainLoss, bestValLoss, optimizer, model = buildModel(args, loadBest=False)

    trainLoader = buildLoader(args, "train")
    valLoader = buildLoader(args, "val")

    logger = SummaryWriter(logdir=os.path.join("runs", args.save))
    for epoch in range(startingEpoch + 1, args.epochs):
        train_results = run(args, epoch, "train", trainLoader, model, optimizer)
        pprint(train_results)
        for metric, value in train_results.items():
            logger.add_scalar("train/{}".format(metric), value, epoch)

        val_results = run(args, epoch, "val", valLoader, model, optimizer)
        pprint(val_results)
        for metric, value in val_results.items():
            logger.add_scalar("val/{}".format(metric), value, epoch)

        # Save Model
        isBestLoss = False
        if val_results["Loss"] < bestValLoss:
            bestTrainLoss, bestValLoss, isBestLoss = train_results["Loss"], val_results["Loss"], True
            with open(os.path.join("logs", args.save, "train_metrics.json"), "w") as f:
                json.dump(train_results, f)
            with open(os.path.join("logs", args.save, "val_metrics.json"), "w") as f:
                json.dump(val_results, f)

        model.saveCheckpoint(
            os.path.join("logs", args.save),
            epoch,
            optimizer,
            bestTrainLoss,
            bestValLoss,
            isBestLoss,
        )


def test(args):
    bestEpoch, bestTrainLoss, bestValLoss, optimizer, model = buildModel(args, loadBest=True)
    print("Train Loss (best model): {:.3f}".format(bestTrainLoss))
    print("Val Loss (best model): {:.3f}".format(bestValLoss))

    testLoader = buildLoader(args, "test")

    test_results = run(args, bestEpoch, "test", testLoader, model, optimizer)
    pprint(test_results)
    with open(os.path.join("logs", args.save, "test_metrics.json"), "w") as f:
        json.dump(test_results, f)

if __name__ == "__main__":

    set_seed(0)
    args = Options()
    print(args)

    train(args.opts)
    args.opts.load = args.opts.save
    test(args.opts)
