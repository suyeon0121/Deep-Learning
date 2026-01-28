import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.num_heads * self.d_k)

        return self.fc(context)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, trg_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, trg_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask)))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x

def make_trg_mask(trg, pad_idx):
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(
    torch.ones((trg_len, trg_len), device=trg.device)).bool()
    trg_sub_mask = trg_sub_mask.unsqueeze(0).unsqueeze(1)
    return trg_pad_mask & trg_sub_mask

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size, 
            d_model=512,
            num_layers=6,
            num_heads=8,
            d_ff=2048,
            dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model

        self.encoder_emb = nn.Embedding(src_vocab_size, d_model)
        self.decoder_emb = nn.Embedding(trg_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.fc_out.weight = self.decoder_emb.weight
    
    def forward(self, src, trg, src_mask, trg_mask):
        enc = self.encoder_emb(src) * math.sqrt(self.d_model)
        enc = self.dropout(self.pos_enc(enc))

        for layer in self.encoder_layers:
            enc = layer(enc, src_mask)

        dec = self.decoder_emb(trg) * math.sqrt(self.d_model)
        dec = self.dropout(self.pos_enc(dec))

        for layer in self.decoder_layers:
            dec = layer(dec, enc, src_mask, trg_mask)
        
        return self.fc_out(dec)
