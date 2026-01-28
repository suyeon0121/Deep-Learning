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
