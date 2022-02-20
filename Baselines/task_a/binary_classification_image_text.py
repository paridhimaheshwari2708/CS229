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
import nltk
from nltk.stem.snowball import SnowballStemmer

from mixed_naive_bayes import MixedNB

from sklearn.metrics import accuracy_score
import torchvision.transforms as T
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

import torch
import pickle
from sklearn.preprocessing import LabelEncoder



with open('image_features.pickle', 'rb') as handle:
    image_features = pickle.load(handle) 

with open('tf_features.pickle', 'rb') as handle:
    text_features = pickle.load(handle) 

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



### For Text Features

def get_image_text_features(img_col, text_col):
	img_features = image_features[img_col]
	txt_features = text_features[text_col]
	print(img_features.shape, txt_features.shape)
	combined_features = np.concatenate((txt_features, img_features), axis = 1)
	print(combined_features.shape)
	# combined_features = [i+t for (i,t) in zip(img_features, txt_features)]
	return combined_features

X_train = get_image_text_features('train_features', 'train')
Y_train = train_df['misogynous']

X_val = get_image_text_features('val_features', 'val')
Y_val = val_df['misogynous']

X_test = get_image_text_features('test_features', 'test')
Y_test = test_df['misogynous']

print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)
print(X_test.shape, Y_test.shape)



print("\n---------------------------------------------------------------------------")
print("-----------------------  LOGISTIC REGRESSION ------------------------------")
print("----------------------------------------------------------------------------")
lr_clf = LogisticRegression(solver='sag').fit(X_train, Y_train)
pred_lr = lr_clf.predict(X_test)
print(classification_report(Y_test, pred_lr, target_names=target_names, digits=3))
acc_score = accuracy_score(test_df[category], pred_lr)
print("Accuracy = {}".format(acc_score))


print("\n--------------------------------------------------------------------")
print("-----------------------   LINEAR SVC   -----------------------------")
print("--------------------------------------------------------------------\n")
lr_svm = LinearSVC().fit(X_train, Y_train)
pred_svm = lr_svm.predict(X_test)
print(classification_report(Y_test, pred_svm, target_names=target_names, digits=3))
acc_score = accuracy_score(test_df[category], pred_svm)
print("Accuracy = {}".format(acc_score))


print("\n--------------------------------------------------------------------")
print("----------------------- SGD CLASSIFIER ------------------------------")
print("--------------------------------------------------------------------\n")
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, Y_train)
pred_clf = sgd_clf.predict(X_test)
print(classification_report(Y_test, pred_clf, target_names=target_names, digits=3))
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


# print("\n--------------------------------------------------------------------")
# print("----------------------- MIXED NAIVE BAYES ------------------------------")
# print("--------------------------------------------------------------------\n")

# X_train = np.array(X_train)
# Y_train = np.array(Y_train)
# label_encoder = LabelEncoder()
# X_train[:,11629] = label_encoder.fit_transform(X_train[:,11629])

# X_test[:,11629] = label_encoder.fit_transform(X_test[:,11629])

# mnb_clf = MixedNB(categorical_features=list(range(11629)))
# mnb_clf.fit(X_train, Y_train)
# pred_clf_nb = mnb_clf.predict(X_test)
# print(classification_report(Y_test, pred_clf_nb, target_names=target_names, digits=3))
# acc_score = accuracy_score(test_df[category], pred_clf_nb)
# print("Accuracy = {}".format(acc_score))








