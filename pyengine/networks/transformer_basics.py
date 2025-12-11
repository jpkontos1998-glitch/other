import einops
import torch.nn as nn
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None, is_causal=False):
        """
        x: [batch, seq, d_model]
        attn_mask: [seq, seq]
        kv_cache: [2, batch, n_head, lookback_len, d_model]
        """
        q = einops.rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_head)
        k = einops.rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.n_head)
        v = einops.rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.n_head)

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, attn_mask=attn_mask, is_causal=is_causal
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mem, is_causal):
        """
        mem: [batch, seq, d_model]
        x: [batch, seq, d_model]
        """
        k = einops.rearrange(self.k_proj(mem), "b t (h d) -> b h t d", h=self.n_head)
        v = einops.rearrange(self.v_proj(mem), "b t (h d) -> b h t d", h=self.n_head)
        q = einops.rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_head)

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout, ff_factor=4):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, n_head)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_factor * d_model)
        self.linear2 = nn.Linear(ff_factor * d_model, d_model)

        self.dropout = dropout

    def forward(self, x, attn_mask=None, is_causal=False):
        x = x + nn.functional.dropout(
            self.mha(self.layer_norm1(x), attn_mask=attn_mask, is_causal=is_causal), p=self.dropout
        )
        x = x + nn.functional.dropout(self._ff_block(self.layer_norm2(x)), p=self.dropout)
        return x

    def _ff_block(self, x):
        x = self.linear2(nn.functional.relu(self.linear1(x)))
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout, ff_factor=4):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadCrossAttention(d_model, n_head)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_factor * d_model)
        self.linear2 = nn.Linear(ff_factor * d_model, d_model)

        self.dropout = dropout

    def forward(self, x, mem, is_causal=False):
        x = x + nn.functional.dropout(
            self.mha(self.layer_norm1(x), mem=mem, is_causal=is_causal), p=self.dropout
        )
        x = x + nn.functional.dropout(self._ff_block(self.layer_norm2(x)), p=self.dropout)
        return x

    def _ff_block(self, x):
        x = self.linear2(nn.functional.relu(self.linear1(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.self_attn_layer = SelfAttentionLayer(d_model, n_head, dropout)
        self.cross_attn_layer = CrossAttentionLayer(d_model, n_head, dropout)

    def forward(self, x, mem):
        x = self.self_attn_layer(x, is_causal=True)
        x = self.cross_attn_layer(x, mem)
        return x
