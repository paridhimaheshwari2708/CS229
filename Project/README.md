# Detecting Misogynistic Content in Multimodal Memes

## Data Preprocessing

```
python preprocess.py
python generate_embeddings.py
```

## Baselines

## Deep Learning Models

For unimodal networks, CNN+LSTM, VQA and MUTAN,
```
python run.py --save {save_folder_name} --mode {mode} --model {model} --image_mode {image_mode} --text_mode {text_mode}
```
where options for various arguments are
- `{mode}` can be either `TaskA` or `TaskB`.
- `{model}` can be `[VQA, MUTAN, Text, Image, ImageText]`
- `{image_mode}` can be `general` for VGG-16 embeddings and `clip` for CLIP pretrained feature extractor.
- `{text_mode}` can be `glove` for GloVe word embeddings or`urban` for Urban Dictionary embeddings.

To use common world knowledge, set `{image_mode}` and `{text_mode}` to `clip` and `urban` respectively. Otherwise, use `general` and `glove`.

## Joint Learning

For unimodal networks, CNN+LSTM, VQA and MUTAN,
```
python run.py --save {save_folder_name} --mode {mode} --model {model} --image_mode {image_mode} --text_mode {text_mode} --hierarchical {hierarchical}
```
where options for various arguments are
- `{mode}` can be either `TaskA` or `TaskB`.
- `{model}` can be `[VQA, MUTAN, ImageText]`
- `{image_mode}` is `general` for VGG-16 embeddings and `clip` for CLIP pretrained feature extractor.
- `{text_mode}` is `glove` for GloVe word embeddings or`urban` for Urban Dictionary embeddings.
- `{hierarchical}` is `all` for multi-task learning and `true` for hierarchical learning.

To use common world knowledge, set `{image_mode}` and `{text_mode}` to `clip` and `urban` respectively. Otherwise, use `general` and `glove`.
