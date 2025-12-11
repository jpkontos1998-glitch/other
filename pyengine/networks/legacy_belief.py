# NOTE: Support for legacy belief model. Ideally should not be modified.

from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
from torch.nn import functional as F

from pyengine.utils import get_pystratego

pystratego = get_pystratego()


def trim_infostate(infostate: torch.Tensor, history_len: int) -> torch.Tensor:
    """Trim the infostate movement planes to `history_len`."""
    board_state = infostate[:, : pystratego.NUM_BOARD_STATE_CHANNELS]
    if history_len == 0:  # Need special case to avoid "-0" indexing
        return board_state
    history_state = infostate[:, pystratego.NUM_BOARD_STATE_CHANNELS :][:, -history_len:]
    return torch.cat([board_state, history_state], dim=1)


# NOTE: copy pasted becasue I want to use the is_causal flag :(
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, is_causal):
        """
        x: [batch, seq, d_model]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=self.n_head).unbind(1)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mem, is_causal):
        """
        mem: [batch, seq, d_model]
        x: [batch, seq, d_model]
        """
        kv = self.kv_proj(mem)
        k, v = einops.rearrange(kv, "b t (k h d) -> b k h t d", k=2, h=self.n_head).unbind(1)
        q = self.q_proj(x)
        q = einops.rearrange(q, "b t (h d) -> b h t d", h=self.n_head)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_head)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)

        self.dropout = dropout

    def forward(self, x, is_causal):
        x = x + nn.functional.dropout(self.mha(self.layer_norm1(x), is_causal), p=self.dropout)
        x = x + nn.functional.dropout(self._ff_block(self.layer_norm2(x)), p=self.dropout)
        return x

    def _ff_block(self, x):
        x = self.linear2(nn.functional.relu(self.linear1(x)))
        return x


class CrossAttnLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mhca = MultiHeadCrossAttention(d_model, n_head)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)

        self.dropout = dropout

    def forward(self, x, mem):
        x = x + nn.functional.dropout(
            self.mhca(self.layer_norm1(x), mem, is_causal=False), p=self.dropout
        )
        x = x + nn.functional.dropout(self._ff_block(self.layer_norm2(x)), p=self.dropout)
        return x

    def _ff_block(self, x):
        x = self.linear2(nn.functional.relu(self.linear1(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.self_attn_layer = SelfAttnLayer(d_model, n_head, dropout)
        self.cross_attn_layer = CrossAttnLayer(d_model, n_head, dropout)

    def forward(self, x, mem):
        x = self.self_attn_layer(x, is_causal=True)
        x = self.cross_attn_layer(x, mem)
        return x


# NOTE: end of copy paste


@dataclass
class ARBeliefConfig:
    depth: int = 6
    num_head: int = 8
    embed_dim: int = 512
    dropout: float = 0
    mask: int = 0
    plane_history_len: int = 86
    decoder_depth: int = 4


class ARBelief(nn.Module):
    def __init__(self, num_piece_type: int, cfg: ARBeliefConfig):
        super().__init__()
        self.cfg = cfg
        self.num_piece_type = num_piece_type

        self.positional_embed_transformer = torch.nn.Linear(100, cfg.embed_dim, bias=False)
        self.positional_embed_decoder = torch.nn.Linear(40, cfg.embed_dim, bias=False)
        self.infostate_embed = nn.Linear(
            pystratego.NUM_BOARD_STATE_CHANNELS + cfg.plane_history_len,
            cfg.embed_dim,
            bias=False,
        )
        self.piece_embed = nn.Linear(num_piece_type, cfg.embed_dim, bias=False)
        self.transformer = nn.ModuleList(
            [SelfAttnLayer(cfg.embed_dim, cfg.num_head, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(cfg.embed_dim, cfg.num_head, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.final_ln_mem = nn.LayerNorm(cfg.embed_dim)
        self.final_ln = nn.LayerNorm(cfg.embed_dim)
        self.final_linear = nn.Linear(cfg.embed_dim, num_piece_type)

    def forward(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        num_moves: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_type_onehot: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
    ) -> torch.Tensor:
        mem = self.encoder_forward(infostate_tensor, unknown_piece_position_onehot)
        return self.decoder_forward(unknown_piece_type_onehot, mem)

    def encoder_forward(
        self,
        infostate_tensor: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
    ) -> torch.Tensor:
        # Initial forward pass
        infostate_tensor = trim_infostate(infostate_tensor, self.cfg.plane_history_len)
        infostate_tensor = infostate_tensor.flatten(-2, -1).permute(0, 2, 1)
        info_tokens = self.infostate_embed(infostate_tensor)
        pos_emb = self.positional_embed_transformer.weight.transpose(0, 1)
        x = info_tokens + pos_emb.unsqueeze(0)
        for layer in self.transformer:
            x = layer(x, is_causal=False)

        # Prep memory for decoder
        piece_pos = unknown_piece_position_onehot.int().argmax(dim=-1)
        piece_info = torch.gather(x, 1, piece_pos.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        active_piece = unknown_piece_position_onehot.any(dim=-1, keepdim=True)
        piece_info.masked_fill_(~active_piece, 0)
        mem = self.final_ln_mem(piece_info)

        return mem

    def decoder_forward(self, unknown_piece_type_onehot, mem):
        # Decoder forward pass
        shifted_piece_type = F.pad(unknown_piece_type_onehot, (0, 0, 1, 0), value=0)[:, :-1]
        z = self.piece_embed(shifted_piece_type.float())
        z = z + self.positional_embed_decoder.weight.transpose(0, 1).unsqueeze(0)
        for layer in self.decoder:
            z = layer(z, mem)
        logit = self.final_linear(self.final_ln(z))

        return logit
