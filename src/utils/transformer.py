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


class HierHiddenTransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0, high_freq=2):
        super().__init__()

        self.low_attention = SelfAttention(emb, heads=heads, mask=mask)
        self.high_attention = SelfAttention(emb, heads=heads, mask=mask)
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

    def forward(self, x_mask_depth):
        x, mask, depth_id = x_mask_depth

        attended = self.low_attention(self.norm1(x), mask)

        x = attended + x

        x = self.do(x)

        fedforward = self.ff(self.norm2(x))

        x = fedforward + x

        x = self.do(x)

        if depth_id % self.high_freq != 0:
            last_token_from_input = x_mask_depth[0][:, -1:, :]  # shape (B, 1, D)
            x = torch.cat([x[:, :-1, :], last_token_from_input], dim=1)
            # x[:, -1, :] = x_mask_depth[0][:, -1, :]

        return x, mask, depth_id + 1


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
        attn = self.tblocks[0].attention.attn_map(tokens, mask)
        return attn


class HRMGrad(nn.Module):

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

        # assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L + z_H + tokens, mask)
        z_H = self.H_level(z_H + z_L, mask)

        x = self.toprobs(z_H.view(b * t, e)).view(b, t, self.num_tokens)

        return x  # , tokens

    def attention_heatmap(self, tokens, mask):
        attn = self.tblocks[0].attention.attn_map(tokens, mask)
        return attn


class HierHiddenTransformer(nn.Module):

    def __init__(self, emb, heads, depth, output_dim, high_freq):
        super().__init__()

        self.num_tokens = output_dim

        # self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                HierHiddenTransformerBlock(
                    emb=emb, heads=heads, mask=False, high_freq=high_freq
                )
            )

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, tokens, mask):

        # tokens = self.token_embedding(x)
        # tokens = torch.cat((x, h), 1)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask, 1))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x  # , tokens

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
