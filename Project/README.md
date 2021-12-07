# Detecting Misogynistic Content in Multimodal Memes

## Setup

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
where options for `{mode}` are `[TaskA, TaskB]`, options for `{model}` are `[VQA, MUTAN, SAN, Text, Image, ImageText]`, options for `{image_mode}` are `[general, clip]` and options for `{text_mode}` are `[glove, urban]`.

To use common world knowledge, set `{image_mode}` to `clip` and `{text_mode}` to `urban`. Otherwise, set `{image_mode}` to `general` and `{text_mode}` to `glove`.

## Joint Learning

For unimodal networks, CNN+LSTM, VQA and MUTAN,
```
python run.py --save {save_folder_name} --mode {mode} --model {model} --image_mode {image_mode} --text_mode {text_mode} --hierarchical {hierarchical}
```
where options for `{mode}` are `[TaskA, TaskB]`, options for `{model}` are `[VQA, MUTAN, SAN, Text, Image, ImageText]`, options for `{image_mode}` are `[general, clip]`, options for `{text_mode}` are `[glove, urban]` and options for `{hierarchical}` are `[all, true]`.

To use common world knowledge, set `{image_mode}` to `clip` and `{text_mode}` to `urban`. Otherwise, set `{image_mode}` to `general` and `{text_mode}` to `glove`.

To train multi-task setting, set `{hierarchical}` to `all`, and use `true` for hierarchical learning.
