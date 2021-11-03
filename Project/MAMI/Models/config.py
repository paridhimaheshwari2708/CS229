import os

DATA_PATH = "/home/ubuntu/CS229/Project/Data"
IMAGE_DIR = os.path.join(DATA_PATH, "images")
VOCABULARY_PATH = os.path.join(DATA_PATH, "vocabulary.npy")
GLOVE_PATH = os.path.join(DATA_PATH, "glove.6B.300d.txt")

IMAGE_DIMENSION = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GLOVE_EMBEDDING_SIZE = 300
GLOVE_OOV_SCALE = 0.6
