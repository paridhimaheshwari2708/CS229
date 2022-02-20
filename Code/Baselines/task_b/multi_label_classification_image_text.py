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


from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import torchvision.transforms as T
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

import torch
import pickle


with open('image_features.pickle', 'rb') as handle:
    image_features = pickle.load(handle) 

with open('tf_idf_features.pickle', 'rb') as handle:
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

# target_names = ['non-misogynistic', 'misogynistic']
categories = ['misogynous', 'shaming','stereotype','objectification','violence']


def read_data_from_csv(path_name):
	df = pd.read_csv(path_name, usecols=['file_name', 'misogynous', text_data_column, 'misogynous', 'shaming','stereotype','objectification','violence'], sep='\t')
	path = image_path+'/'
	df['image_path'] = path + df['file_name']
	return df

## Reading data
train_df = read_data_from_csv(csv_path_train)
val_df = read_data_from_csv(csv_path_val)
test_df = read_data_from_csv(csv_path_test)


def get_image_text_features(img_col, text_col):
    img_features = image_features[img_col]
    txt_features = text_features[text_col]
    print(img_features.shape, txt_features.shape)
    combined_features = np.concatenate((img_features, txt_features), axis = 1)
    print(combined_features.shape)
    # combined_features = [i+t for (i,t) in zip(img_features, txt_features)]
    return combined_features

X_train = get_image_text_features('train_features', 'train')
Y_train = train_df[categories]

X_val = get_image_text_features('val_features', 'val')
Y_val = val_df[categories]

X_test = get_image_text_features('test_features', 'test')
Y_test = test_df[categories]

print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)
print(X_test.shape, Y_test.shape)


print("---------------------------------------------------------------------------")
print("-----------------------  LOGISTIC REGRESSION ------------------------------")
print("---------------------------------------------------------------------------")
LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])

prediction_labels = []
gt_labels = []
accuracy = []
for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    LogReg_pipeline.fit(X_train, train_df[category])
    # compute the testing accuracy
    prediction = LogReg_pipeline.predict(X_test)
    prediction_labels.append(prediction)
    gt_labels.append(test_df[category])
    acc_score = accuracy_score(test_df[category], prediction)
    accuracy.append(acc_score)
    print('Test accuracy is {}'.format(acc_score))

pred_format = np.array(prediction_labels).T
gt_format = np.array(gt_labels).T
print(classification_report(gt_format, pred_format, target_names=categories, digits=3))
print("Accuracy = {}".format(np.mean(accuracy)))


print("--------------------------------------------------------------------")
print("-----------------------   LINEAR SVC   -----------------------------")
print("--------------------------------------------------------------------")
SVC_pipeline = Pipeline([('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),])

prediction_labels = []
gt_labels = []
accuracy = []
for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    SVC_pipeline.fit(X_train, train_df[category])
    # compute the testing accuracy
    prediction = SVC_pipeline.predict(X_test)
    prediction_labels.append(prediction)
    gt_labels.append(test_df[category])
    acc_score = accuracy_score(test_df[category], prediction)
    accuracy.append(acc_score)
    print('Test accuracy is {}'.format(acc_score))

pred_format = np.array(prediction_labels).T
gt_format = np.array(gt_labels).T
print(classification_report(gt_format, pred_format, target_names=categories, digits=3))
print("Accuracy = {}".format(np.mean(accuracy)))

# print("--------------------------------------------------------------------")
print("----------------------- SGD CLASSIFIER ------------------------------")
print("--------------------------------------------------------------------")
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, Y_train)
pred_clf = sgd_clf.predict(X_test)
print(classification_report(Y_test, pred_clf, target_names=target_names))

print("--------------------------------------------------------------------")
print("-----------------------   NAIVE BAYES   -----------------------------")
print("--------------------------------------------------------------------")
NB_pipeline = Pipeline([('clf', OneVsRestClassifier(GaussianNB(), n_jobs=1)),])

prediction_labels = []
gt_labels = []
accuracy = []
for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    NB_pipeline.fit(X_train, train_df[category])
    # compute the testing accuracy
    prediction = NB_pipeline.predict(X_test)
    prediction_labels.append(prediction)
    gt_labels.append(test_df[category])
    acc_score = accuracy_score(test_df[category], prediction)
    accuracy.append(acc_score)
    print('Test accuracy is {}'.format(acc_score))

pred_format = np.array(prediction_labels).T
gt_format = np.array(gt_labels).T
print(classification_report(gt_format, pred_format, target_names=categories, digits=3))
print("Accuracy = {}".format(np.mean(accuracy)))








