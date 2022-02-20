import os
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
import torchvision.transforms as T
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

import torch
import pickle


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


with open('image_features.pickle', 'rb') as handle:
    image_features = pickle.load(handle) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = '/dfs/user/paridhi/CS229/Project/Data'

csv_path_train = os.path.join(DATA_PATH, 'split_train.csv')
csv_path_val = os.path.join(DATA_PATH, 'split_val.csv')
csv_path_test = os.path.join(DATA_PATH, 'split_test.csv')

image_path = os.path.join(DATA_PATH, 'images_training')
IMAGE_DIMENSION = (224, 224)

text_data_column = 'Text Normalized'
# text_data_column = 'Text Transcription'

target_names = ['non-misogynistic', 'misogynistic']
category = ['misogynous']

def read_data_from_csv(path_name):
	df = pd.read_csv(path_name, usecols=['file_name', 'misogynous', text_data_column], sep='\t')
	path = image_path+'/'
	df['image_path'] = path + df['file_name']
	return df

## Reading data
train_df = read_data_from_csv(csv_path_train)
val_df = read_data_from_csv(csv_path_val)
test_df = read_data_from_csv(csv_path_test)

X_train = image_features['train_features']
Y_train = train_df['misogynous']

X_val = image_features['val_features']
Y_val = val_df['misogynous']

X_test = image_features['test_features']
Y_test = test_df['misogynous']

print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)
print(X_test.shape, Y_test.shape)



print("\n---------------------------------------------------------------------------")
print("-----------------------  LOGISTIC REGRESSION ------------------------------")
print("----------------------------------------------------------------------------")
lr_clf = LogisticRegression(solver='sag').fit(X_train, Y_train)
pred_lr = lr_clf.predict(X_test)
print(classification_report(Y_test, pred_lr, target_names=target_names, digits=4))
acc_score = accuracy_score(test_df[category], pred_lr)
print("Accuracy = {}".format(acc_score))


print("\n--------------------------------------------------------------------")
print("-----------------------   LINEAR SVC   -----------------------------")
print("--------------------------------------------------------------------\n")
lr_svm = LinearSVC().fit(X_train, Y_train)
pred_svm = lr_svm.predict(X_test)
print(classification_report(Y_test, pred_svm, target_names=target_names, digits=4))
acc_score = accuracy_score(test_df[category], pred_svm)
print("Accuracy = {}".format(acc_score))


print("\n--------------------------------------------------------------------")
print("----------------------- SGD CLASSIFIER ------------------------------")
print("--------------------------------------------------------------------\n")
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, Y_train)
pred_clf = sgd_clf.predict(X_test)
print(classification_report(Y_test, pred_clf, target_names=target_names, digits=4))
acc_score = accuracy_score(test_df[category], pred_clf)
print("Accuracy = {}".format(acc_score))



print("\n--------------------------------------------------------------------")
print("----------------------- GAUSSIAN NAIVE BAYES ------------------------------")
print("--------------------------------------------------------------------\n")
gnb_clf = GaussianNB()
gnb_clf.fit(X_train, Y_train)
pred_clf_nb = gnb_clf.predict(X_test)
print(classification_report(Y_test, pred_clf_nb, target_names=target_names, digits=3))
acc_score = accuracy_score(test_df[category], pred_clf_nb)
print("Accuracy = {}".format(acc_score))

print("\n--------------------------------------------------------------------")
print("----------------------- DECISION TREE ------------------------------")
print("--------------------------------------------------------------------\n")
gnb_clf = DecisionTreeClassifier()
gnb_clf.fit(X_train, Y_train)
pred_clf_nb = gnb_clf.predict(X_test)
print(classification_report(Y_test, pred_clf_nb, target_names=target_names, digits=3))
acc_score = accuracy_score(test_df[category], pred_clf_nb)
print("Accuracy = {}".format(acc_score))








