from fastai.text import *
import sentencepiece as spm

class LanguageTokenizer(BaseTokenizer):
    def __init__(self, sp_model):
        # self.lang = lang
        self.sp = spm.SentencePieceProcessor()
        model_path = sp_model
        self.sp.Load(str(model_path))

    def tokenizer(self, t: str) -> List[str]:
        return self.sp.EncodeAsPieces(t)

    def numericalize(self, t: str) -> List[int]:
        return self.sp.EncodeAsIds(t)

    def textify(self, ids: List[int]) -> str:
        return (''.join([self.sp.IdToPiece(id).replace('▁', ' ') for id in ids])).strip()



class BengaliTokenizer(LanguageTokenizer):
    def __init__(self, sp_model):
        LanguageTokenizer.__init__(self, sp_model)
