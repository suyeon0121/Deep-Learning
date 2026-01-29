## Encoder-Decoder Transformer
### Overview
Attention Is All You Need 논문을 기반으로 Transformer Encoder-Decoder 구조 구현을 통해 공부
RNN 없이 Self-Attention만으로 시퀀스 간 의존성을 학습한다. 
 
### Key Idea
- 순서 정보는 Positional Encoding으로 보완
- Self_Attention으로 모든 토큰 간 관계를 병렬적으로 학습
- Multi-Head Attention을 통해 다양한 표현 공간에서 관계 포착
- Encoder-Decoder 구조로 Seq2Seq 문제 해결
 
### Model Architecture
**Embedding**
- Source / Target Embedding
- Scaling: sqrt(d_model)
- Positional Encoding 추가

**Encoder**
- Multi-Head Self-Attention
- Feed Forward Network
- Residual Connection + LayerNorm
- 총 num_layers 반복

**Decoder**
- Masked Multi-Head Self-Attention
- Encoder-Decoder Cross Attention
- Feed Forward Network
- Residual Connection + LayerNorm
- 총 num_layers 반복

### Output
- Linear Projection -> Target Vocabulary
- Embedding weight tying 적용

### Special Implementation Points
- Padding Mask + Subsequent Mask 분리 처리
- Attention score scaling(√d_k)
- Dropout으로 과적합 방지
- Embedding과 Output Layer 가중치 공유

---

## Note
### Module 설계 및 초기화 
- PyTorch 모델은 반드시 nn.Module을 상속해야 하며, super().init__() 호출을 통해 하위 모듈과 파라미터가 정상적으로 등족되도록 했다.
- 해당 초기화 과정이 누락될 경우 optimizer, device 이동, autograd 동작에 문제가 발생할 수 있다.

### Dropout 설정
- nn.Dropout의 인자는 확률(p) 이므로, d_model과 같은 차원 값이 아닌 dropout 비율을 명시적으로 전달
- 잘못된 인자 전달 시 런타임 에러 또는 학습 불안정이 발생할 수 있다.

### Decoder Causal Mask 적용
- Decoder self-attention에는 subsequent mask(causal mask)를 적용하여 현재 시점 이후의 토큰 정보를 참조하지 못하도록 제한
- 이를 통해 auto-regressive decoding 조건을 만족하도록 구현

### Attention Mask Shape & Device 관리
- Attention score의 shape에 맞게 mask를 (batch, 1, seq_len, seq_len) 형태로 구성하여 multi-head 차원으로 자연스럽게 broadcast 되도록 했다.
- mask tensor는 입력 tensor와 동일한 device를 사용하도록 명시하여 CPU/GPU mismatch 문제를 방지

### Embedding Weight Tying
- Decoder embedding과 output projection layer의 가중치를 공유하여 파라미터 수를 줄이고 일반화 성능 향상을 도모했다.
- 이는 Transformer 논문에서 사용되는 표준적인 구현 기법
