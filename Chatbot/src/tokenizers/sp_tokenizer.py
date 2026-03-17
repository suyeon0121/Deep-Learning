import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text):
        # 가장 표준적이고 안전한 방식인 encode_as_ids를 사용
        return self.sp.encode_as_ids(text)

    def decode(self, tokens):
        # 토큰 ID 리스트(또는 텐서)를 텍스트로 변환
        if hasattr(tokens, "tolist"): # 파이토치 텐서인 경우 리스트로 변환
            tokens = tokens.tolist()
        return self.sp.decode(tokens)

    # 특수 토큰 ID를 반환하는 메서드들 (train.yaml 설정값과 일치)
    def pad_id(self):
        return self.sp.pad_id() # 보통 0

    def bos_id(self):
        return self.sp.bos_id() # 보통 1

    def eos_id(self):
        return self.sp.eos_id() # 보통 2

    def vocab_size(self):
        return self.sp.get_piece_size()