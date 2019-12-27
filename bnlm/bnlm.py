import os
import torch
from fastai.text import *
import numpy as np
import pickle
import pathlib
from math import ceil
import sentencepiece as spm
from zipfile import ZipFile
from bnlm.tokenizer import LanguageTokenizer
from bnlm.utils import cos_sim
from bnlm.downloads import download_file_from_google_drive

BASE_PATH = os.getcwd()
sp_model = os.path.join(BASE_PATH, 'model/bn_spm.model')

class BengaliTokenizer(BaseTokenizer):
    def __init__(self, lang:str):
        self.lang = lang
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model)
        
    def tokenizer(self, t:str) -> List[str]:
        return self.sp.EncodeAsPieces(t)

def download_models():
    id = "1bhF5GV-pN_rrSVqe_inE4GgCkOnFKVL_"
    directory = "model"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = directory+'/'+'models.zip'
    print("Downloading Models...")
    print("It will take sometimes..")
    download_file_from_google_drive(id, file_name)
    zf = ZipFile(file_name, 'r')
    zf.extractall(directory)
    zf.close()
    os.remove(file_name)
    print("Download completed")

def predict_n_words(input: str, n_words: int, model_path: str, randomness=0.8):
    defaults.device = torch.device('cpu')
    learn = load_learner(model_path)
    output = learn.predict(input, n_words, randomness)
    output = input + (''.join(output.replace(input, '').split(' '))).replace('▁', ' ')
    return output

def get_sentence_encoding(input: str, model_path: str, sp_model: str):
    tok = LanguageTokenizer(sp_model)
    token_ids = tok.numericalize(input)
    defaults.device = torch.device('cpu')
    learn = load_learner(model_path)
    encoder = learn.model[0]
    encoder.reset()
    kk0 = encoder(Tensor([token_ids]).to(torch.int64))
    return np.array(kk0[0][-1][0][-1])

def get_sentence_similarity(sen1:str, sen2:str, model_path:str, sp_model:str, cmp: Callable = cos_sim):
    enc1 = get_sentence_encoding(sen1, model_path, sp_model)
    enc2 = get_sentence_encoding(sen2, model_path, sp_model)
    return cmp(enc1, enc2)

def get_embedding_vectors(input:str, model_path:str, sp_model:str):
    tok = LanguageTokenizer(sp_model)
    token_ids = tok.numericalize(input)
    defaults.device = torch.device('cpu')
    learn = load_learner(model_path)
    encoder = get_model(learn.model)[0]
    encoder.reset()
    embeddings = encoder.state_dict()['encoder.weight']
    embeddings = np.array(embeddings)
    embedding_vectors = []
    for token in token_ids:
        embedding_vectors.append(embeddings[token])
    return embedding_vectors

def get_similar_sentences(sen: str, no_of_variations: int, model_path:str, sp_model:str):
    tok = LanguageTokenizer(sp_model)
    token_ids = tok.numericalize(sen)
    embedding_vectors = get_embedding_vectors(sen, model_path, sp_model)
    defaults.device = torch.device('cpu')
    learn = load_learner(model_path)
    encoder = get_model(learn.model)[0]
    encoder.reset()
    embeddings = encoder.state_dict()['encoder.weight']
    embeddings = np.array(embeddings)
    scores = []
    for word_vec in embedding_vectors:
        scores.append([cos_sim(word_vec, embdg) for embdg in embeddings])
    word_ids = [np.argpartition(-np.array(score), no_of_variations)[:no_of_variations] for score in scores]
    new_token_ids = []
    no_of_vars_per_token = ceil(no_of_variations/len(token_ids))*3
    for i in range(len(token_ids)):
        word_ids_list = word_ids[i].tolist()
        word_ids_list.remove(token_ids[i])
        for j in range(no_of_vars_per_token):
            new_token_ids.append(token_ids[:i] + word_ids_list[j:j+1] + token_ids[i+1:])
    new_sens = [tok.textify(tok_id) for tok_id in new_token_ids]
    while sen in new_sens:
        new_sens.remove(sen)
    sen_with_sim_score = [(new_sen, get_sentence_similarity(sen, new_sen, model_path, sp_model)) for new_sen in new_sens]
    sen_with_sim_score.sort(key=lambda x: x[1], reverse=True)
    new_sens = [sen for sen, _ in sen_with_sim_score]
    return new_sens[:no_of_variations]

# if __name__ == "__main__":
#     model_path = '../model'
#     sp_model = "/home/sagor/bnlp/model/bn_spm.model"
#     output = predict_n_words("আমি বাজারে", 3, model_path)
#     print("Word Prediction: ", output)
    # encoding = get_sentence_encoding("আমি ভাত খাই।", model_path, sp_model)
    # print("sentence encoding is: ", encoding)
    # embed = get_embedding_vectors("আমি ভাত খাই।", model_path, sp_model)
    # print("sentence embedding is : ", embed)
    # sim = get_sentence_similarity("আমি ভাত খাই।", "আমি ভাত খাই।", model_path, sp_model)
    # print("similarity is: ", sim)
    # sen_pred = get_similar_sentences("আমি ভাত খাই।", 3, model_path, sp_model)
    # print(sen_pred)

 

