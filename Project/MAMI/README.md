
# Detecting Misogynistic Content in Multimodal Memes

## Data Preprocessing

Download the following:
- Meme data from [Multimedia Automatic Misogyny Identification (MAMI) Challenge](https://competitions.codalab.org/competitions/34175).
- Pretrained [GloVe vectors](https://nlp.stanford.edu/projects/glove/).
- Pretrained [Urban Dictionary embeddings](http://smash.inf.ed.ac.uk/ud-embeddings/).

Run the following commands for preprocessing the meme data, glove embeddings and urban dictionary embeddings.
```
python preprocess.py
python generate_embeddings.py
```

## Baselines

Refer to the [README for baselines](Baselines/README.md)

## Deep Learning Models

For unimodal networks, CNN+LSTM, VQA and MUTAN,
```
cd Models/
python run.py --save {save_folder_name} --mode {mode} --model {model} --image_mode {image_mode} --text_mode {text_mode}
```
where options for various arguments are
- `{mode}` can be either `TaskA` or `TaskB`.
- `{model}` can be `[VQA, MUTAN, Text, Image, ImageText]`
- `{image_mode}` can be `general` for VGG-16 embeddings and `clip` for CLIP pretrained feature extractor.
- `{text_mode}` can be `glove` for GloVe word embeddings or`urban` for Urban Dictionary embeddings.

To use common world knowledge, set `{image_mode}` and `{text_mode}` to `clip` and `urban` respectively. Otherwise, use `general` and `glove`.

For BERT-based models, refer to the following [README](Models_BERT/README.md).

## Joint Learning

For unimodal networks, CNN+LSTM, VQA and MUTAN,
```
cd HierarchicalModels/
python run.py --save {save_folder_name} --mode {mode} --model {model} --image_mode {image_mode} --text_mode {text_mode} --hierarchical {hierarchical}
```
where options for various arguments are
- `{mode}` can be either `TaskA` or `TaskB`.
- `{model}` can be `[VQA, MUTAN, ImageText]`
- `{image_mode}` is `general` for VGG-16 embeddings and `clip` for CLIP pretrained feature extractor.
- `{text_mode}` is `glove` for GloVe word embeddings or`urban` for Urban Dictionary embeddings.
- `{hierarchical}` is `all` for multi-task learning and `true` for hierarchical learning.

To use common world knowledge, set `{image_mode}` and `{text_mode}` to `clip` and `urban` respectively. Otherwise, use `general` and `glove`.

For BERT-based models, refer to the following [README](HierarchicalModels_BERT/README.md).
