# Bengal Language Model
[![Build Status](https://travis-ci.org/sagorbrur/bnlm.svg?branch=master)](https://travis-ci.org/sagorbrur/bnlm)
[![Documentation Status](https://readthedocs.org/projects/bnlm/badge/?version=latest)](https://bnlm.readthedocs.io/en/latest/?badge=latest)
[![pypi version](https://img.shields.io/pypi/v/bnlm)](https://pypi.org/project/bnlm/)
[![python version](https://img.shields.io/badge/python-3.6%7C3.7-brightgreen)](https://pypi.org/project/bnlm/)


Bengali language model is build with fastai's [ULMFit](https://arxiv.org/abs/1801.06146) and ready for `prediction` and `classfication` task.


NB: 
* This tool mostly followed [inltk](https://github.com/goru001/inltk)
* We separated `Bengali` part with better evaluation results

# Installation

`pip install bnlm`


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
sentence_1 = "আমি ভাত খাই।"
sentence_2 = "আমি ভাত খাই।"
sim = get_sentence_similarity(sentence_1, sentence_2, model_path, sp_model)
print("similarity is: ", sim)

```

## Get Simillar Sentences
```py
from bnlm.bnlm import BengaliTokenizer
from bnlm.bnlm import get_similar_sentences

model_path = 'model'
sp_model = "model/bn_spm.model"

input_sentence = "আমি ভাত খাই।"
sen_pred = get_similar_sentences(input_sentence, 3, model_path, sp_model)
print(sen_pred)


```


## Classification
```upcomming```

# Training
To train with your own corpus follow [this](https://github.com/sagorbrur/Bengali-Language-Model) repository
