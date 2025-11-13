# aeayn_transformer.py
# A full "Attention Is All You Need" encoder-decoder Transformer (PyTorch)

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Positional Encoding (sinusoidal)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [L, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, L, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ----------------------------
# Multi-Head Attention (from scratch, tensor-friendly)
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor):
        # x: [B, L, d_model] -> [B, H, L, d_head]
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        return x

    def _combine_heads(self, x: torch.Tensor):
        # x: [B, H, L, d_head] -> [B, L, d_model]
        B, H, L, Dh = x.shape
        x = x.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return x

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        q, k, v: [B, L, d_model]
        attn_mask: [Lq, Lk] or [B, Lq, Lk] with True/-inf masking semantics (see below)
        key_padding_mask: [B, Lk] with True for PAD positions
        """
        B, Lq, _ = q.shape
        Lk = k.size(1)

        q = self._split_heads(self.q_proj(q))
        k = self._split_heads(self.k_proj(k))
        v = self._split_heads(self.v_proj(v))
        # [B, H, L, d_head]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, Lq, Lk]

        # attn_mask: True/1 == mask; or a tensor with -inf at masked positions
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask.unsqueeze(1), float("-inf"))
            else:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(1)  # broadcast

        # key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: True means pad → mask out
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, Lq, d_head]
        out = self._combine_heads(out)  # [B, Lq, d_model]
        return self.o_proj(out)


# ----------------------------
# Position-wise FeedForward
# ----------------------------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ----------------------------
# Encoder/Decoder Layers
# ----------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        attn_out = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN
        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Masked self-attention (causal)
        attn_out = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross attention over encoder memory
        attn_out = self.cross_attn(
            x, memory, memory, attn_mask=None, key_padding_mask=memory_key_padding_mask
        )
        x = self.norm2(x + self.dropout(attn_out))

        # FFN
        ff = self.ffn(x)
        x = self.norm3(x + self.dropout(ff))
        return x


# ----------------------------
# Encoder / Decoder stacks
# ----------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, src_tokens: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        """
        src_tokens: [B, S]  (token ids)
        src_key_padding_mask: [B, S], True for PAD
        """
        x = self.embed(src_tokens)  # [B, S, d_model]
        x = self.pos(x)
        src_mask = None  # encoder is non-causal
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x  # [B, S, d_model]


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )
        self.d_model = d_model

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device):
        # True where we want to mask (upper triangle)
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
        return mask  # [T, T], True=masked

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        tgt_tokens: [B, T]
        memory: [B, S, d_model] from Encoder
        *_key_padding_mask: [B, len], True for PAD
        """
        B, T = tgt_tokens.shape
        x = self.embed(tgt_tokens)
        x = self.pos(x)

        tgt_mask = self._generate_square_subsequent_mask(T, x.device)  # [T, T]
        # broadcast to [B, T, T] inside attention via mask semantics

        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x  # [B, T, d_model]


# ----------------------------
# Full Seq2Seq Transformer
# ----------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_idx: int = 0,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, d_model, nhead, num_encoder_layers, d_ff, dropout, max_len, padding_idx
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, nhead, num_decoder_layers, d_ff, dropout, max_len, padding_idx
        )
        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)

        if tie_embeddings:
            # Tie output projection with decoder embedding weights (common in NMT/LMs)
            self.lm_head.weight = self.decoder.embed.weight

        self.padding_idx = padding_idx
        self.d_model = d_model

    def make_padding_mask(self, tokens: torch.Tensor):
        # True where PAD
        return tokens.eq(self.padding_idx)

    def forward(self, src_tokens: torch.Tensor, tgt_tokens_in: torch.Tensor):
        """
        src_tokens: [B, S]
        tgt_tokens_in: [B, T]  (typically BOS + target[:T-1])
        Returns: logits [B, T, tgt_vocab]
        """
        src_key_padding_mask = self.make_padding_mask(src_tokens)  # [B, S]
        tgt_key_padding_mask = self.make_padding_mask(tgt_tokens_in)  # [B, T]

        memory = self.encoder(src_tokens, src_key_padding_mask=src_key_padding_mask)
        dec = self.decoder(
            tgt_tokens_in,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.lm_head(dec)
        return logits  # [B, T, Vt]

    @torch.no_grad()
    def generate(
        self,
        src_tokens: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 128,
        greedy: bool = True,
    ):
        """
        Greedy decoding (for simplicity). Returns [B, T_out]
        """
        device = src_tokens.device
        B = src_tokens.size(0)
        src_key_padding_mask = self.make_padding_mask(src_tokens)
        memory = self.encoder(src_tokens, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            dec_out = self.decoder(
                ys,
                memory,
                tgt_key_padding_mask=self.make_padding_mask(ys),
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = self.lm_head(dec_out[:, -1:, :])  # last step, [B, 1, V]
            if greedy:
                next_token = torch.argmax(logits.squeeze(1), dim=-1)  # [B]
            else:
                probs = F.softmax(logits.squeeze(1), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

            finished |= next_token.eq(eos_id)
            if finished.all():
                break

        return ys  # includes BOS and generated tokens
        

# ----------------------------
# Loss helper (Label Smoothing Cross Entropy)
# ----------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes: int, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred: [B, T, V]
        target: [B, T]  (gold ids aligned with pred steps)
        """
        pred = pred.view(-1, pred.size(-1))  # [B*T, V]
        target = target.view(-1)             # [B*T]

        # mask out ignore positions
        ignore = target.eq(self.ignore_index)
        target = target.masked_fill(ignore, 0)

        # one-hot with smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[ignore] = 0.0

        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(true_dist * log_probs).sum(dim=1)
        loss = loss.masked_fill(ignore, 0.0)
        denom = (~ignore).sum().clamp_min(1)
        return loss.sum() / denom


# ----------------------------
# Minimal training step example
# ----------------------------
def training_step_example(model, batch, optimizer, loss_fn):
    """
    batch must provide:
      - src: [B, S]
      - tgt_in: [B, T]  (shifted right with BOS)
      - tgt_out: [B, T] (gold, ends with EOS + PADs)
    """
    model.train()
    src, tgt_in, tgt_out = batch["src"], batch["tgt_in"], batch["tgt_out"]
    logits = model(src, tgt_in)  # [B, T, Vt]
    loss = loss_fn(logits, tgt_out)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()
