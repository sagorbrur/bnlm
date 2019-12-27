import unittest
import bnlm
from bnlm.bnlm import BengaliTokenizer
from bnlm.bnlm import download_models
from bnlm.bnlm import predict_n_words, get_sentence_encoding, get_embedding_vectors
from bnlm.bnlm import get_sentence_similarity, get_similar_sentences

class TestBNLP(unittest.TestCase):

    def dumm_test(self):
        version = bnlm.__version__
        self.assertEqual('1.0.0', version)

if __name__ == '__main__':
    unittest.main()
