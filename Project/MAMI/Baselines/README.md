# Unimodal and Multimodal Baseline Experiments on Misogynistic Classification

Experiments on binary and multi-label classification using baseline models like Naive Bayes, SVM, Logistic Regression

Image Features:
* ``preprocess_image_features.py`` - Extract the image features and store them into a pickle file

task_a:
- Binary Classification into misogynistic and not-misogynistic memes
* ``./task_a/binary_classification_text.ipynb`` - Unimodal binary classification of memes using textual BOW features
* ``./task_a/binary_classification_images.py`` - Unimodal binary classification of memes using image features extracted from VGG 
* ``./task_a/binary_classification_image_text.py`` - Multimodal binary classification of memes using both text and image features

task_b:
- Multi-Label Classification into 5 classes
* ``./task_b/multi_label_classification_text.ipynb`` - Multi-label classification of memes using textual BOW features (Unimodal)
* ``./task_b/binary_classification_images.py`` - Multi-label classification of memes using image features extracted from VGG (Unimodal)
* ``./task_b/binary_classification_image_text.py`` - Multi-label classification of memes using both text and image features 