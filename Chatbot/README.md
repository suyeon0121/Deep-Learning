## Chatbot
### Overview
- Transformer 기반 Encoder-Decoder 구조를 구현하여 텍스트 기반 대화를 생성하는 챗봇을 구축

### Model Architecture
Encoer
- Embedding + Positional Encoding
- Multi-head Self Attention
- Feed Forward Network(GELU)
- Residual Connection + LayerNorm
<br/>
Decoder
- Masked Self Attention(causal mask)
- Encoder-Decoder Attention (cross attention)
- Feed Forward Network
- Residual Connection + LayerNorm

<br/>
- Positional Encoding
   - Transformer는 순서 정보를 직접 다루지 않기 때문에, sin/cos 기반 positional encoding을 추가하여 토큰의 위치 정보를 반영한다.
- Multi-head Attention
   - 여러 attention head를 사용하여 문법, 의미, 장거리 의존성 등 다양한 관계를 병렬적으로 학습한다.
- Masking
   - Padding Mask: 의미 없는 padding token의 attention을 차단
   - Causal Mask: 미래 토큰 접근을 제한하여 autoregressive 구조 유지
- Weight Tying
   - Decoder embedding과 output projection을 공유하여 파라미터 수 감소 및 일반화 성능 향상
   - Input Embedding: 단어(ID)를 보고 벡터로 변환 / Output Projection: 계산된 벡트를 보고 다시 어떤 단어인지 이름을 찾음
     - 즉, 입력과 출력을 하나로 통일하여 모델의 무게는 줄이고 성능은 높이

<br/>
### Tokenizer
- sentencePiece(Unigram)
- vacabulary size: 8000

### Dataset
- 질문, 답변 형식의 대화 데이터
```
src: [BOS] question [EOS]
trg: [BOS] answer [EOS]
```

### Training
- Loss: CrossEntoryLoss
- Optimizer: Adam(lr=1e-4)
- Batch Size: 16
- Epochs: 50
<br/>
- Teacher Forcing을 사용하여 다음 토근을 예측
<br/>

```
input  : trg[:, :-1]
target : trg[:, 1:]
```
- Gradient Clipping(exploding gradient 방지)
- Padding/Causal mask 적용
<br/>

### Inference
Autoregressive Generation
```
<BOS> → token 생성 → 반복 → <EOS>
```
- 이전까지 생성된 토큰을 입력으로 사용하여 다음 토큰을 순차적으로 생

### Result
- 학습이 진행됨에 따라 loss 감소
- 기본적인 문장 구조 생성
- 데이터 규모가 작아 표현 다양성 제한
<br/>

### Error
1. 데이터 오염 및 토크나이저 mismatch
- 챗봇이 빈칸을 출력하거나 특정 기호만 반복함
- 원인: 실제 대화가 아닌 CSV 헤더 위주로 학습되어 한글 단어가 사전에 등록되지 않았다.
- 해결: 기존 데이터를 삭제하고 한글 대화 기반 데이터를 다시 재학습시켰다.

2. 모델-토크나이저 동기화
- 사전을 업데이트 했음에도 계속 엉뚱한 답변을 출력함
- 원인: 사전의 내용은 새롭게 바뀌었으나 모델 가중치는 과거의 사전을 기억하고 있어 불일치 발생
- 해결: shutil.copy로 경로를 강제로 동기화하고, 다시 새로운 사전으로 처음부터 학습 시작
     
3. 논리 오류
- 챗봇이 질문에 답하는 것이 아닌 질문을 그대로 따라 출력하거나 무의미한 문장을 나열
- 원인: 기존 데이터가 질문-답변이 아닌 단순 문장 나열로 구성되어 모델이 다음 토큰 예측 시 질문과 답변을 구별하지 못함
- 해결: 데이터의 구조를 [질문] [TAB] [답변] 구조로 설계를 하고 모델이 질문 뒤에 오는 답변을 타겟으로 인식하도록 함

4. 자원 최적화
- 데이터 증량 후 학습 중 CPU 과열 및 제대로 된 답변 생성 못함
- 원인: 노트북 환경으로 고려하지 않은 과도한 데이터 학습
- 해결: Epoch 수 줄이고 batch size 조정, 데이터 5000개만 학습





