import sentencepiece as spm
import os
import shutil

# 1. 설정 (경로 및 파라미터)
CORPUS_FILE = "corpus.txt"
MODEL_PREFIX = "tokenizer"
TARGET_DIR = "src/tokenizers"
VOCAB_SIZE = 8000

def rebuild():
    # 2. 기존 사전 파일 삭제 (충돌 방지)
    if os.path.exists(f"{MODEL_PREFIX}.model"):
        os.remove(f"{MODEL_PREFIX}.model")
        os.remove(f"{MODEL_PREFIX}.vocab")

    # 3. 사전 학습 시작
    print(f"{CORPUS_FILE} 기반으로 새로운 사전(Vocab: {VOCAB_SIZE}) 학습 중...")
    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE, 
        model_prefix=MODEL_PREFIX, 
        vocab_size=VOCAB_SIZE, 
        model_type="unigram",
        user_defined_symbols=["[PAD]", "[BOS]", "[EOS]"],
        character_coverage=1.0,
        hard_vocab_limit=False
    )

    # 4. 지정된 경로로 사전 파일 이동/복사
    if os.path.exists(TARGET_DIR):
        shutil.copy(f"{MODEL_PREFIX}.model", os.path.join(TARGET_DIR, f"{MODEL_PREFIX}.model"))
        print(f"사전을 {TARGET_DIR}로 성공적으로 배달했습니다.")
    else:
        print("목표 폴더가 없어 현재 폴더에 사전을 유지합니다.")

if __name__ == "__main__":
    rebuild()