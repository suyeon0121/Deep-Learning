import torch

def make_src_mask(src, pad_idx):
    # 만약 pad_idx가 숫자가 아니라면(device 객체 등), 강제로 0을 사용
    if not isinstance(pad_idx, (int, float)):
        actual_pad = 0
    else:
        actual_pad = pad_idx

    # src가 텐서가 아닐 경우를 대비해 변환
    src = torch.as_tensor(src)
    
    # torch.ne() 대신 직관적인 != 연산자를 사용하고, 비교 대상을 확실히 숫자로 고정
    # Encoder용
    mask = (src != int(actual_pad)).int()
    return mask.unsqueeze(1).unsqueeze(2)

def make_trg_mask(trg, pad_idx):
    # trg_mask도 똑같이 보호 로직을 추가
    if not isinstance(pad_idx, (int, float)):
        actual_pad = 0
    else:
        actual_pad = pad_idx

    trg = torch.as_tensor(trg)
    
    # 1. 패딩 마스크
    trg_pad_mask = (trg != int(actual_pad)).int().unsqueeze(1).unsqueeze(2)
    
    # 2. 순방향 마스크 (Look-ahead mask)
    trg_len = trg.size(1)
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).int()
    
    # 두 마스크 결합
    return trg_pad_mask & trg_sub_mask