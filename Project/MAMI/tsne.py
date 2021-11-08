import pickle
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

with open('tf_idf_features.pickle', 'rb') as f:
    tfidf = pickle.load(f)

labels = {}
for subset in ['train', 'val', 'test']:
    labels[subset] = pd.read_csv('split_{}.csv'.format(subset), sep='\t', usecols=['misogynous']).to_numpy().squeeze()

# tfidf_tsne = TSNE(n_components=2, n_jobs=4, verbose=2).fit_transform(tfidf['test'])
tfidf_pca = PCA(n_components=2).fit_transform(tfidf['test'])
print(tfidf_tsne.shape, tfidf_pca.shape)
plt.scatter(tfidf_pca[:,0], tfidf_pca[:,1], c=labels['test'])
plt.show()
