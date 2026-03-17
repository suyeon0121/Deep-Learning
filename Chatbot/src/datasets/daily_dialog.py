import torch
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DailyDialogDataset(Dataset):
    def __init__(self, tokenizer, file_path="corpus.txt"):
        self.tokenizer = tokenizer
        self.examples = []
        
        # 1. 데이터 로드 및 분리 (콤마 기준 1회 분할)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue 
                
                # '질문,답변' 형식을 안전하게 쪼개기
                parts = line.split(",", 1) if "," in line else [line, ""]
                self.examples.append(parts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 2. 텍스트 추출 및 텐서 변환 ([BOS]=1, [EOS]=2)
        question, answer = self.examples[idx] if len(self.examples[idx]) == 2 else (self.examples[idx][0], "...")
        
        src = torch.tensor([1] + self.tokenizer.encode(str(question)) + [2])
        trg = torch.tensor([1] + self.tokenizer.encode(str(answer)) + [2])
        return src, trg

# 3. 배치를 같은 길이로 맞추는 패딩 함수
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
    return src_batch, trg_batch