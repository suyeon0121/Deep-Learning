import torch

from src.models.transformer import Transformer
from src.tokenizers.sp_tokenizer import SentencePieceTokenizer
from src.utils.mask import make_src_mask

tokenizer=SentencePieceTokenizer("tokenizer.model")

model=Transformer(8000,8000)

model.load_state_dict(torch.load("chatbot.pt"))

model.eval()

while True:

    text=input("You: ")

    tokens=tokenizer.encode(text)

    src=torch.tensor(tokens).unsqueeze(0)

    mask=make_src_mask(src,0)

    print("Bot: ...")