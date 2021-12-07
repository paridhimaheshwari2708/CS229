import os
import numpy as np

DATA_PATH = "/dfs/user/paridhi/CS229/Project/Data"
GLOVE_RAW = os.path.join(DATA_PATH, "glove.6B.300d.txt")
GLOVE_PATH = os.path.join(DATA_PATH, "glove.6B.300d.npy")
URBAN_RAW = os.path.join(DATA_PATH, "ud_basic.vec")
URBAN_PATH = os.path.join(DATA_PATH, "ud_basic.npy")

def process_glove():
	glove = {}
	with open(GLOVE_RAW, 'rb') as f:
		for line in f:
			line = line.decode().split()
			word = line[0]
			vec = np.array(line[1:]).astype(float)
			glove[word] = vec
	np.save(GLOVE_PATH, glove)

def process_urban_dictionary():
	urban = {}
	with open(URBAN_RAW, 'rb') as f:
		next(f) # ignoring first line (header)
		for line in f:
			line = line.decode().split()
			word = line[0]
			vec = np.array(line[1:]).astype(float)
			urban[word] = vec
	np.save(URBAN_PATH, urban)

if __name__ == '__main__':

	# Processing glove
	process_glove()

	# Processing urban dictionary embeddings
	process_urban_dictionary()