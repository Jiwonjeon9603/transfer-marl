import torch.nn as nn
import torch.nn.functional as F
import torch
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):

        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        if (
            self.mask
        ):  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float("-inf"), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities
        # self.attn_map = dot.detach().cpu()
        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

    def attn_map(self, x, mask):

        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        if self.mask:
            mask_(dot, maskval=float("-inf"), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)

        return dot


class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask

        attended = self.attention(x, mask)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, mask


class TransformerBlockPreNorm(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0, high_freq=2):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask
        self.high_freq = high_freq

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask

        attended = self.attention(self.norm1(x), mask)

        x = attended + x

        x = self.do(x)

        fedforward = self.ff(self.norm2(x))

        x = fedforward + x

        x = self.do(x)

        return x, mask


class TSFFNTransformerBlock(nn.Module):

    def __init__(
        self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0, n_hist_tokens=2
    ):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask
        self.n_hist_tokens = n_hist_tokens

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        # FF for observation tokens
        self.ff_obs = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

        # FF for history tokens
        self.ff_hist = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask

        # Attention
        attended = self.attention(x, mask)
        x = self.do(self.norm1(x + attended))

        # Split into observation vs history tokens
        obs, hist = x[:, : -self.n_hist_tokens], x[:, -self.n_hist_tokens :]

        # Apply different FFNs
        obs = self.ff_obs(obs)
        hist = self.ff_hist(hist)

        # Concatenate back
        fedforward = torch.cat([obs, hist], dim=1)

        # Residual + dropout
        x = self.norm2(x + fedforward)
        x = self.do(x)

        return x, mask


class Transformer(nn.Module):

    def __init__(self, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        # self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, tokens, mask):

        # tokens = self.token_embedding(x)
        # tokens = torch.cat((x, h), 1)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x  # , tokens

    def attention_heatmap(self, tokens, mask):
        attn = self.tblocks[0].attention.attn_map(tokens, mask)
        return attn




class HRMTransformerBlock(nn.Module):
    
    def __init__(self, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        # self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

    def forward(self, tokens, mask):

        # tokens = self.token_embedding(x)
        # tokens = torch.cat((x, h), 1)

        x, mask = self.tblocks((tokens, mask))

        return x  # , tokens

    def attention_heatmap(self, tokens, mask):
        attn = self.tblocks[0].attention.attn_map(tokens, mask)
        return attn

class HRM(nn.Module):
    
    def __init__(self, emb, heads, depth, output_dim, h_cycles=2, l_cycles=2):
        super().__init__()

        self.num_tokens = output_dim
        self.L_level = HRMTransformerBlock(emb, heads, depth, output_dim)
        self.H_level = HRMTransformerBlock(emb, heads, depth, output_dim)
        self.H_cycles = h_cycles
        self.L_cycles = l_cycles

        self.toprobs = nn.Linear(emb, output_dim)
        # self.token_embedding = nn.Linear(input_dim, emb)

    def forward(self, tokens, mask):

        # tokens = self.token_embedding(x)
        # tokens = torch.cat((x, h), 1)
        b, t, e = tokens.size()

        with torch.no_grad():
            z_H = torch.zeros_like(tokens)
            z_L = torch.zeros_like(tokens)
            # z_H = trunc_normal_init_(
            #     torch.empty((b, t, e), dtype=tokens.dtype, device=device), std=1
            # )
            # z_L = trunc_normal_init_(
            #     torch.empty((b, t, e), dtype=tokens.dtype, device=device), std=1
            # )
            for itr_step in range(1, self.H_cycles * self.L_cycles):
                z_L = self.L_level(z_L + z_H + tokens, mask)
                if itr_step % self.H_cycles == 0:
                    z_H = self.H_level(z_H + z_L, mask)

            # for _H_step in range(self.H_cycles):
            #     for _L_step in range(self.L_cycles):
            #         if not (
            #             (_H_step == self.H_cycles - 1)
            #             and (_L_step == self.L_cycles - 1)
            #         ):
            #             z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

            #     if not (_H_step == self.config.H_cycles - 1):
            #         z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L + z_H + tokens, mask)
        z_H = self.H_level(z_H + z_L, mask)

        x = self.toprobs(z_H.view(b * t, e)).view(b, t, self.num_tokens)

        return x  # , tokens

    def attention_heatmap(self, tokens, mask):
        b, t, e = tokens.size()
        with torch.no_grad():
            z_H = torch.zeros_like(tokens)
            z_L = torch.zeros_like(tokens)

            for itr_step in range(1, self.H_cycles * self.L_cycles):
                z_L = self.L_level(z_L + z_H + tokens, mask)

                Low_hidden_1 = self.L_level.attention_heatmap(z_L + z_H + tokens, mask)

                if itr_step % self.H_cycles == 0:
                    z_H = self.H_level(z_H + z_L, mask)

                    # 1-step grad
            z_L = self.L_level(z_L + z_H + tokens, mask)
            Low_hidden_2 = self.L_level.attention_heatmap(z_L + z_H + tokens, mask)

            z_H = self.H_level(z_H + z_L, mask)
            High_hidden = self.H_level.attention_heatmap(z_H + z_L, mask)

        return Low_hidden_1, Low_hidden_2, High_hidden
    
class HRMTSFFNTransformerBlock(nn.Module):
    
    def __init__(self, emb, heads, depth, output_dim, n_hist_tokens):
        super().__init__()

        self.num_tokens = output_dim

        # self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(TSFFNTransformerBlock(emb=emb, heads=heads, mask=False, n_hist_tokens=n_hist_tokens))

        self.tblocks = nn.Sequential(*tblocks)

    def forward(self, tokens, mask):

        # tokens = self.token_embedding(x)
        # tokens = torch.cat((x, h), 1)

        x, mask = self.tblocks((tokens, mask))

        return x  # , tokens

    def attention_heatmap(self, tokens, mask):
        attn = self.tblocks[0].attention.attn_map(tokens, mask)
        return attn

class HRMTSFFN(nn.Module):
    
    def __init__(self, emb, heads, depth, output_dim, h_cycles, l_cycles, n_hist_tokens):
        super().__init__()

        self.num_tokens = output_dim
        self.L_level = HRMTSFFNTransformerBlock(emb, heads, depth, output_dim, n_hist_tokens)
        self.H_level = HRMTSFFNTransformerBlock(emb, heads, depth, output_dim, n_hist_tokens)
        self.H_cycles = h_cycles
        self.L_cycles = l_cycles

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, tokens, mask):
        b, t, e = tokens.size()

        with torch.no_grad():
            z_H = torch.zeros_like(tokens)
            z_L = torch.zeros_like(tokens)

            for itr_step in range(1, self.H_cycles * self.L_cycles):
                z_L = self.L_level(z_L + z_H + tokens, mask)
                if itr_step % self.H_cycles == 0:
                    z_H = self.H_level(z_H + z_L, mask)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L + z_H + tokens, mask)
        z_H = self.H_level(z_H + z_L, mask)

        x = self.toprobs(z_H.view(b * t, e)).view(b, t, self.num_tokens)

        return x  # , tokens

    def attention_heatmap(self, tokens, mask):
        b, t, e = tokens.size()
        with torch.no_grad():
            z_H = torch.zeros_like(tokens)
            z_L = torch.zeros_like(tokens)

            for itr_step in range(1, self.H_cycles * self.L_cycles):
                z_L = self.L_level(z_L + z_H + tokens, mask)

                Low_hidden_1 = self.L_level.attention_heatmap(z_L + z_H + tokens, mask)

                if itr_step % self.H_cycles == 0:
                    z_H = self.H_level(z_H + z_L, mask)

                    # 1-step grad
            z_L = self.L_level(z_L + z_H + tokens, mask)
            Low_hidden_2 = self.L_level.attention_heatmap(z_L + z_H + tokens, mask)

            z_H = self.H_level(z_H + z_L, mask)
            High_hidden = self.H_level.attention_heatmap(z_H + z_L, mask)

        return Low_hidden_1, Low_hidden_2, High_hidden
    



class HRM_wo_h(nn.Module):

    def __init__(self, emb, heads, depth, output_dim, n_cycles=2):
        super().__init__()

        self.num_tokens = output_dim
        self.block = HRMTransformerBlock(emb, heads, depth, output_dim)
        # self.H_level = HRMTransformerBlock(emb, heads, depth, output_dim)
        self.n_cycles = n_cycles

        self.toprobs = nn.Linear(emb, output_dim)
        # self.token_embedding = nn.Linear(input_dim, emb)

    def forward(self, tokens, mask):

        # tokens = self.token_embedding(x)
        # tokens = torch.cat((x, h), 1)
        b, t, e = tokens.size()

        z = torch.zeros_like(tokens)
        for itr_step in range(self.n_cycles):
            z = self.block(z + tokens, mask)

        x = self.toprobs(z.view(b * t, e)).view(b, t, self.num_tokens)

        return x  # , tokens

    def attention_heatmap(self, tokens, mask):
        b, t, e = tokens.size()
        with torch.no_grad():
            z = torch.zeros_like(tokens)
            for itr_Step in range(self.n_cycles):
                z = self.block(z + tokens, mask)
                attnetion = self.block.attention_heatmap(z + tokens, mask)

        return attnetion


class TSFFNTransformer(nn.Module):

    def __init__(self, emb, heads, depth, output_dim, n_hist_tokens):
        super().__init__()

        self.num_tokens = output_dim

        # self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TSFFNTransformerBlock(
                    emb=emb, heads=heads, mask=False, n_hist_tokens=n_hist_tokens
                )
            )

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, tokens, mask):

        # tokens = self.token_embedding(x)
        # tokens = torch.cat((x, h), 1)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x  # , tokens

    def attention_heatmap(self, tokens, mask):
        attn = self.tblocks[0].attention.attn_map(tokens, mask)
        return attn

class TSFFNInputTransformer(nn.Module):
    def __init__(self, emb, heads, depth, output_dim, n_hist_tokens):
        super().__init__()
        self.num_tokens = output_dim

        # 여러 transformer block을 ModuleList에 저장
        self.tblocks = nn.ModuleList([
            TSFFNTransformerBlock(
                emb=emb, heads=heads, mask=False, n_hist_tokens=n_hist_tokens
            )
            for _ in range(depth)
        ])

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, tokens, mask):
        b, t, e = tokens.size()

        # z_0 초기화
        with torch.no_grad():
            z = torch.zeros_like(tokens)

        # 각 block마다 tokens + z를 입력으로 넣고 z 갱신
        for block in self.tblocks:
            z = block(tokens + z, mask)

        # 최종 출력 projection
        out = self.toprobs(z.view(b * t, e)).view(b, t, self.num_tokens)

        return out

    def attention_heatmap(self, tokens, mask):
        attn = self.tblocks[0].attention.attn_map(tokens, mask)
        return attn


def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


def trunc_normal_init_(
    tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0
):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor



