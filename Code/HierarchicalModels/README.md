
# Joint Learning Paradigms for Detecting Misogynistic Content in Multimodal Memes

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
