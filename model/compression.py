import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch, seq, dim = x.shape
        x = x.view(batch, seq, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        batch, heads, seq, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq, heads * head_dim)

    def forward(self, queries, context):
        q = self._split_heads(self.q_proj(queries))
        k = self._split_heads(self.k_proj(context))
        v = self._split_heads(self.v_proj(context))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        return out


class CompressionTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)

        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, queries, context):
        q_norm = self.self_attn_norm(queries)
        self_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self_out

        q_norm = self.cross_attn_norm(queries)
        cross_out = self.cross_attn(q_norm, context)
        queries = queries + cross_out

        q_norm = self.ffn_norm(queries)
        ffn_out = self.ffn(q_norm)
        queries = queries + ffn_out
        return queries


class CompressionTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, n_compress_tokens, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.n_compress_tokens = n_compress_tokens
        self.max_context_length = 4096

        self.compress_queries = nn.Parameter(torch.randn(1, n_compress_tokens, d_model) * 0.02)
        self.context_pos_embedding = nn.Parameter(torch.randn(1, self.max_context_length, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                CompressionTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, context):
        context_len = context.shape[1]
        if context_len > self.max_context_length:
            raise ValueError(
                f"context length {context_len} exceeds max_context_length {self.max_context_length}"
            )

        context = context + self.context_pos_embedding[:, :context_len, :]
        queries = self.compress_queries.expand(context.shape[0], -1, -1)

        for layer in self.layers:
            queries = layer(queries, context)

        return self.final_norm(queries)
