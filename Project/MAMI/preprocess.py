import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

DATA_PATH = "/home/ubuntu/CS229/Project/Data"
csv_path = os.path.join(DATA_PATH, 'training.csv')
MAX_TEXT_LENGTH = 50

def normalize_text(text):
	text = text.lower()
	text = re.sub('[^a-zA-Z]', ' ', text)
	text = re.sub(r'\s+', ' ', text)
	# # Removing Stop Words
	# words = [w for w in text.split(' ') if w not in stopwords.words('english')]
	# text = ' '.join(words)
	# text = re.sub(r'\s+', ' ', text)
	return text.strip()

if __name__ == '__main__':

	df = pd.read_csv(csv_path, delimiter='\t')

	# Preprocessing input text and creating vocabulary
	unique_words = ['unk']
	cols = []
	for i, row in df.iterrows():
		text = df.loc[i]['Text Transcription']
		text = normalize_text(text)
		cols.append(text)
		unique_words.extend(text.split())
	df['Text Normalized'] = cols
	unique_words = list(set(unique_words))
	unique_words_map = {unique_words[i]:i for i in range(len(unique_words))}
	print('Vocabulary size: {}'.format(len(unique_words)))

	# Getting word indices
	cols = []
	for i, row in df.iterrows():
		text = df.loc[i]['Text Normalized']
		words = text.split()
		word_ids = [unique_words_map.get(tmp) for tmp in words[:MAX_TEXT_LENGTH]]
		if len(word_ids) < MAX_TEXT_LENGTH:
			word_ids = word_ids + [unique_words_map.get('unk')] * (MAX_TEXT_LENGTH - len(word_ids))
		cols.append(word_ids)
	df['Text Indices'] = cols

	# Splitting data into train / val / test sets
	n = len(df)
	df = df.sample(frac = 1, random_state=42)
	train, val, test = np.split(df, [int(0.7*n), int(0.9*n)])

	# Saving data
	np.save(os.path.join(DATA_PATH, 'vocabulary.npy'), unique_words)
	train.to_csv(os.path.join(DATA_PATH, 'split_train.csv'), index=False, sep ='\t')
	val.to_csv(os.path.join(DATA_PATH, 'split_val.csv'), index=False, sep ='\t')
	test.to_csv(os.path.join(DATA_PATH, 'split_test.csv'), index=False, sep ='\t')
