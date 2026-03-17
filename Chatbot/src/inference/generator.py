import sys, os, torch
import torch.nn.functional as F
sys.path.append(os.getcwd())

from src.models.transformer import Transformer
from src.tokenizers.sp_tokenizer import SentencePieceTokenizer
from src.utils.mask import make_src_mask, make_trg_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "src/tokenizers/tokenizer.model"
WEIGHT_PATH = "chatbot.pt"

def generate():
    # 1. 모델 및 토크나이저 로드
    tokenizer = SentencePieceTokenizer(MODEL_PATH)
    model = Transformer(src_vocab_size=8000, trg_vocab_size=8000).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 2. 입력 데이터 준비
    input_text = sys.argv[1] if len(sys.argv) > 1 else "안녕"
    src_indices = [1] + tokenizer.encode(input_text) + [2] # [BOS] ... [EOS]
    src = torch.LongTensor(src_indices).unsqueeze(0).to(DEVICE)
    src_mask = make_src_mask(src, 0)
    
    trg_indices = [1] # [BOS]로 시작

    # 3. 자동 완성 루프 (Autoregressive Inference)
    with torch.no_grad():
        for _ in range(50):
            trg = torch.LongTensor(trg_indices).unsqueeze(0).to(DEVICE)
            trg_mask = make_trg_mask(trg, 0)
            
            output = model(src, trg, src_mask, trg_mask)
            logits = output[0, -1, :] 
            
            # --- 샘플링 전략 (핵심 로직) ---
            for token_id in set(trg_indices): logits[token_id] /= 1.5 # 중복 방지
            logits = logits / 0.8 # Temperature
            
            top_k = 50
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float('Inf') # Top-k 필터링
            
            probs = F.softmax(logits, dim=-1) # 확률 변환
            next_token = torch.multinomial(probs, num_samples=1).item() # 샘플링
            # ------------------------------

            if next_token == 2: break # [EOS] 만나면 중단
            trg_indices.append(next_token)
            
    # 4. 답변 정제 (Post-processing)
    full_response = tokenizer.decode(trg_indices)

    if ',' in full_response:
        parts = [p.strip() for p in full_response.split(',')]
        # 질문이 포함되어 있다면 두 번째 조각을, 아니면 첫 번째 조각을 답변으로 선택
        clean_response = parts[1] if len(parts) >= 2 and (parts[0] in input_text) else parts[0]
    else:
        clean_response = full_response.strip()

    # 5. 최종 출력
    print(f"\n나: {input_text}")
    print(f"챗봇: {clean_response.replace(input_text, '').strip().rstrip(',')}")

if __name__ == "__main__":
    generate()