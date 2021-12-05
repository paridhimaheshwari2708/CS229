import os
import sys
import numpy as np
import matplotlib.pyplot as plt

labels = ['misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
scores = np.array([[0.84, 0.85, 0.84],
                    [0.43, 0.57, 0.49],
                    [0.57, 0.69, 0.62],
                    [0.61, 0.66, 0.63],
                    [0.48, 0.35, 0.40]])

num_classes, num_metrics = scores.shape
ind = np.arange(num_metrics) 
width = 0.15

plt.figure()
for i in range(num_classes):
    curr_metric = scores[i, :]
    plt.bar(ind + i*width, curr_metric, width, label=labels[i])
plt.xticks(ind + 2*width, ['Precision', 'Recall', 'F1'])
plt.xlabel("Metric")
plt.legend()
plt.show()
plt.savefig('class_metrics.png')
