from tqdm import tqdm
import torch
from src.utils.mask import make_src_mask, make_trg_mask

class Trainer:
    def __init__(self, model, loader, optimizer, criterion, pad, device="cpu"):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device 
        self.pad = pad

    def train_epoch(self):
        self.model.train()
        total = 0

        pbar = tqdm(self.loader, desc="Training")

        for src, trg in pbar:
            # 1. 데이터를 해당 장치(GPU/MPS/CPU)로 이동
            src = src.to(self.device)
            trg = trg.to(self.device)

            # 2. 마스크 생성 (입력값에 맞춰 전달)
            # trg[:, :-1]은 마지막 토큰(EOS)을 제외한 입력용 타겟
            src_mask = make_src_mask(src, self.pad)
            trg_mask = make_trg_mask(trg[:, :-1], self.pad)

            # 3. 모델 순전파 (Forward)
            out = self.model(src, trg[:, :-1], src_mask, trg_mask)

            # 4. 손실 계산
            # out: [batch, seq_len-1, vocab_size] -> [batch * (seq_len-1), vocab_size]
            # trg[:, 1:]: [batch, seq_len-1] -> [batch * (seq_len-1)] (첫 BOS를 제외한 정답)
            loss = self.criterion(
                out.reshape(-1, out.size(-1)),
                trg[:, 1:].reshape(-1)
            )

            # 5. 역전파 및 최적화
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (노트북 환경에서 학습 불안정 방지용)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            total += loss.item()
            pbar.set_postfix(loss=loss.item()) # 실시간 loss 출력

        return total / len(self.loader)
