# Bengal Language Model
[![Build Status](https://travis-ci.org/sagorbrur/bnlm.svg?branch=master)](https://travis-ci.org/sagorbrur/bnlm)
[![Documentation Status](https://readthedocs.org/projects/bnlm/badge/?version=latest)](https://bnlm.readthedocs.io/en/latest/?badge=latest)
[![pypi version](https://img.shields.io/pypi/v/bnlm)](https://pypi.org/project/bnlm/)
[![python version](https://img.shields.io/badge/python-3.6%7C3.7-brightgreen)](https://pypi.org/project/bnlm/)


Bengali language model is build with fastai's [ULMFit](https://arxiv.org/abs/1801.06146) and ready for `prediction` and `classfication` task.

# Contents
- [Installation](#installation)
- [Bengali Next Word Prediction](#predict-n-words)
- [Bengali Sentence Encoding](#get-sentence-encoding)
- [Bengali Embedding Vectors](#get-embedding-vectors)
- [Bengali Sentence Similarity](#sentence-similarity)
- [Bengali Simillar Sentences](#get-simillar-sentences)


NB: 
* **This tool mostly followed [inltk](https://github.com/goru001/inltk)**
* We separated `Bengali` part with better evaluation results

# Installation

`pip install bnlm`

## Dependencies
* use pytorch >=1.0.0 and <=1.3.0


# Evaluation Result

## Language Model
* Accuracy 48.26% on validation dataset
* Perplexity: ~22.79


# Features and API

## Download pretrained Model
To start, first download pretrained Language Model and Sentencepiece model

```py
from bnlm.bnlm import download_models

download_models()

```
## Predict N Words
`predict_n_words` take three parameter as input:
- input_sen(Your incomplete input text)
- N(Number of word for prediction)
- model_path(Pretrained model path)

```py
from bnlm.bnlm import BengaliTokenizer
from bnlm.bnlm import predict_n_words
model_path = 'model'
input_sen = "আমি বাজারে"
output = predict_n_words(input_sen, 3, model_path)
print("Word Prediction: ", output)

```

## Get Sentence Encoding
```py
from bnlm.bnlm import BengaliTokenizer
from bnlm.bnlm import get_sentence_encoding
model_path = 'model'
sp_model = "model/bn_spm.model"
input_sentence = "আমি ভাত খাই।"
encoding = get_sentence_encoding(input_sentence, model_path, sp_model)
print("sentence encoding is: ", encoding)

```

## Get Embedding Vectors
```py
from bnlm.bnlm import BengaliTokenizer
from bnlm.bnlm import get_embedding_vectors
model_path = 'model'
sp_model = "model/bn_spm.model"
input_sentence = "আমি ভাত খাই।"
embed = get_embedding_vectors(input_sentence, model_path, sp_model)
print("sentence embedding is : ", embed)


```


## Sentence Similarity
```py
from bnlm.bnlm import BengaliTokenizer
from bnlm.bnlm import get_sentence_similarity
model_path = 'model'
sp_model = "model/bn_spm.model"
sentence_1 = "সে খুব করে কথা বলে।"
sentence_2 = "তার কথা খুবেই মিষ্টি।"
sim = get_sentence_similarity(sentence_1, sentence_2, model_path, sp_model)
print("Similarity is: %0.2f"%sim)

# Output:  0.72

```

## Get Simillar Sentences
`get_similar_sentences` take four parameter
- input sentence
- N(Number of sentence you want to predict)
- model_path(Pretrained Model Path)
- sp_model(pretrained sentencepiece model)

```py
from bnlm.bnlm import BengaliTokenizer
from bnlm.bnlm import get_similar_sentences

model_path = 'model'
sp_model = "model/bn_spm.model"

input_sentence = "আমি বাংলায় গান গাই।"
sen_pred = get_similar_sentences(input_sentence, 3, model_path, sp_model)
print(sen_pred)
# output: ['আমি বাংলায় গান গাই ।', 'আমি ইংরেজিতে গান গাই।', 'আমি বাংলায় গানও গাই।']

```


## Classification
```upcomming```

# Training
To train with your own corpus follow [this](https://github.com/sagorbrur/Bengali-Language-Model) repository

# Contributor
[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/images/0)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/links/0)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/images/1)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/links/1)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/images/2)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/links/2)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/images/3)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/links/3)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/images/4)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/links/4)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/images/5)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/links/5)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/images/6)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/links/6)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/images/7)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlm/links/7)
