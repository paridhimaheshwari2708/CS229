import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.style
matplotlib.style.use('tableau-colorblind10')
plt.rc('axes', axisbelow=True)

labels = ['Misogynous', 'Shaming', 'Stereotype', 'Objectification', 'Violence']
scores = np.array([[0.846, 0.84, 0.85, 0.84],
                    [0.87, 0.43, 0.57, 0.49],
                    [0.768, 0.57, 0.69, 0.62],
                    [0.843, 0.61, 0.66, 0.63],
                    [0.899, 0.48, 0.35, 0.40]])


num_classes, num_metrics = scores.shape
ind = np.arange(num_metrics) 
width = 0.15

plt.figure(figsize=(7, 4.8))
plt.grid()
for i in range(num_classes):
    curr_metric = scores[i, :]
    plt.bar(ind + i*width, curr_metric, width, label=labels[i], edgecolor='black', linewidth=0.5)
plt.xticks(ind + 2*width, ['Accuracy', 'Precision', 'Recall', 'F1'])
plt.legend()
plt.show()
plt.savefig('class_metrics.png')
