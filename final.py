

import os
import re
import numpy as np
import pandas as pd

DATA_PATH = "/dfs/user/paridhi/CS229/Project/Data"
csv_path = os.path.join(DATA_PATH, 'final.csv')
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

	# Loading input data
	df = pd.read_csv(csv_path, delimiter='\t')

	# Loading vocabulary
	unique_words = np.load(os.path.join(DATA_PATH, 'vocabulary.npy'))
	unique_words_map = {unique_words[i]:i for i in range(len(unique_words))}
	unk_token = unique_words_map.get('unk')

	# Preprocessing input text
	cols = []
	all_words = []
	for i, row in df.iterrows():
		text = df.loc[i]['Text Transcription']
		text = normalize_text(text)
		cols.append(text)
		all_words.extend(text.split())
	df['Text Normalized'] = cols
	all_words = set(all_words)
	print('Size of vocabulary: {}'.format(len(unique_words)))
	print('Out of vocabulary words: {}'.format(len(all_words.difference(unique_words))))

	# Getting word indices
	cols = []
	total_count, unk_count = 0, 0
	for i, row in df.iterrows():
		text = df.loc[i]['Text Normalized']
		words = text.split()
		word_ids = [unique_words_map.get(tmp, unk_token) for tmp in words[:MAX_TEXT_LENGTH]]
		total_count += len(word_ids)
		unk_count += word_ids.count(unk_token)
		if len(word_ids) < MAX_TEXT_LENGTH:
			word_ids = word_ids + [unk_token] * (MAX_TEXT_LENGTH - len(word_ids))
		cols.append(word_ids)
	df['Text Indices'] = cols
	print('Fraction of words not in vocabulary: {}'.format(unk_count/total_count))

	# Saving data
	df.to_csv(os.path.join(DATA_PATH, 'split_final.csv'), index=False, sep ='\t')
