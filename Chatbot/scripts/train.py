import torch
import os, sys
from torch.utils.data import DataLoader

# 환경 설정
torch.set_num_threads(4)
sys.path.append(os.getcwd())

from src.models.transformer import Transformer
from src.datasets.daily_dialog import DailyDialogDataset, collate_fn
from src.tokenizers.sp_tokenizer import SentencePieceTokenizer
from src.training.trainer import Trainer

def train():
    device = torch.device("cpu")
    
    # 1. 데이터 준비
    tokenizer = SentencePieceTokenizer("src/tokenizers/tokenizer.model")
    dataset = DailyDialogDataset(tokenizer, file_path="corpus.txt")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # 2. 모델 및 학습 도구 설정
    model = Transformer(src_vocab_size=8000, trg_vocab_size=8000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # 패딩 무시

    # 3. 학습 실행
    trainer = Trainer(model, loader, optimizer, criterion, device)
    print("학습 시작")
    
    for epoch in range(50):
        loss = trainer.train_epoch()
        print(f"Epoch [{epoch+1}/50] | Loss: {loss:.4f}")

    # 4. 결과 저장
    torch.save(model.state_dict(), "chatbot.pt")
    print("학습 완료 및 모델 저장 성공!")

if __name__ == "__main__":
    train()