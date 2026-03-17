import torch
import torch.nn as nn
from src.utils.mask import make_trg_mask

def beam_search(model, src, src_mask, beam_size, max_len, bos_idx, eos_idx, device):
    model.eval()
    
    # 1. 초기 상태 설정: (토큰 리스트, 누적 로그 확률)
    # 확률 곱셈 대신 로그 확률 덧셈을 사용하여 Underflow를 방지합니다.
    sequences = [([bos_idx], 0.0)] 

    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            
            for seq, score in sequences:
                # 마지막 토큰이 EOS라면 더 이상 생성하지 않고 후보에 유지
                if seq[-1] == eos_idx:
                    all_candidates.append((seq, score))
                    continue
                
                # 2. 모델 입력 준비
                trg = torch.tensor([seq]).to(device)
                trg_mask = make_trg_mask(trg, 0) # 디코더용 마스크 생성
                
                # 3. 예측 및 로그 확률 계산
                out = model(src, trg, src_mask, trg_mask)
                log_probs = torch.log_softmax(out[:, -1], dim=-1) # log_softmax 권장
                
                # 4. Top-K 후보 추출
                topk_probs, topk_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    next_token = topk_indices[0][i].item()
                    next_score = topk_probs[0][i].item()
                    all_candidates.append((seq + [next_token], score + next_score))

            # 5. 모든 후보 중 상위 beam_size개만 남김 (높은 점수 순)
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_size]
            
            # 모든 후보가 EOS로 끝났다면 조기 종료
            if all(s[0][-1] == eos_idx for s in sequences):
                break

    # 최적의 시퀀스 반환 (BOS 제외)
    return sequences[0][0][1:]