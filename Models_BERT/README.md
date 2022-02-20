# Multimodal BERT models for multimodal classification with Text and Images

General architecture:

* Text representation: Last BERT 786 dimensional hidden vectors (Taking average of all hidden vectors or taking hidden vector associated with CLS token)
* Image representation: VGG16 4096 dimensional vector feature

Both text and image features are concatenated and passed through:

* MLP which outputs prediction classes.
* Multimodal Gated Layer (based on https://arxiv.org/abs/1702.01992) which weights relevance of each modality and combines them to output prediction classes

Datasets used include:

* MAMI dataset for detecting Misogyny in memes (Both Binary and Multi-Label setting)
* Hateful memes detection from Facebook Challenge
* Multimodal IMDb (used plot of movie as text and poster of movie as image)

Commands to train the models:

```
python main.py --dataset mami_dataset --image_mode clip --image_feature_size 512 --model GatedAverageBERT --name gated_avg_clip
```
* ``--dataset`` : ``mami_dataset`` for binary classification, ``mami_multi_dataset`` for multi-label classification
* ``--model``: Can set to ``ConcatBERT, AverageBERT, GatedAverageBERT``
* For using clip embeddings, set ``--image_mode`` to ``clip`` and  ``--image_feature_size`` to  ``512``
* ``--name``: Name of the model you want to save


Commands to test the models:

```
python main.py --dataset mami_dataset --model_mode test --image_mode clip --image_feature_size 512 --model AverageBERT --best_model_cpt avg_bert_clip
```
* Set ``--model_model`` to ``test`` for testing the model
* ``--dataset`` : ``mami_dataset`` for binary classification, ``mami_multi_dataset`` for multi-label classification
* ``--model``: Can set to ``ConcatBERT, AverageBERT, GatedAverageBERT``
* For using clip embeddings, set ``--image_mode`` to ``clip`` and  ``--image_feature_size`` to  ``512``
* ``best_model_cpt``: Name of the model you want to load

