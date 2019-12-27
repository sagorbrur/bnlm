from bnlm.bnlm import BengaliTokenizer
from bnlm.bnlm import download_models
from bnlm.bnlm import predict_n_words, get_sentence_encoding, get_embedding_vectors
from bnlm.bnlm import get_sentence_similarity, get_similar_sentences



if __name__ == "__main__":
    # download_models()
    model_path = 'model'
    sp_model = "model/bn_spm.model"
    output = predict_n_words("আমি বাজারে", 3, model_path)
    print("Word Prediction: ", output)
    # encoding = get_sentence_encoding("আমি ভাত খাই।", model_path, sp_model)
    # print("sentence encoding is: ", encoding)
    # embed = get_embedding_vectors("আমি ভাত খাই।", model_path, sp_model)
    # print("sentence embedding is : ", embed)
    # sim = get_sentence_similarity("আমি ভাত খাই।", "আমি ভাত খাই।", model_path, sp_model)
    # print("similarity is: ", sim)
    # sen_pred = get_similar_sentences("আমি ভাত খাই।", 3, model_path, sp_model)
    # print(sen_pred)