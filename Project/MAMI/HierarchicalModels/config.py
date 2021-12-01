import os

DATA_PATH = "/dfs/user/paridhi/CS229/Project/Data"
IMAGE_DIR = os.path.join(DATA_PATH, "images_training")
VOCABULARY_PATH = os.path.join(DATA_PATH, "vocabulary.npy")
GLOVE_PATH = os.path.join(DATA_PATH, "glove.6B.300d.npy")
URBAN_PATH = os.path.join(DATA_PATH, "ud_basic.npy")

IMAGE_DIMENSION = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EMBEDDING_SIZE = 300
OOV_SCALE = 0.6

TOTAL_SAMPLES = 10000
CLASS_DISTRIBUTION = [5000, 1274, 2810, 2202, 953]
CLASS_POS_WEIGHTS = [1.0, 6.849293563579278, 2.5587188612099645, 3.541326067211626, 9.49317943336831]

ALPHA = 1
