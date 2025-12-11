# NOTE: Support for legacy RL model. Ideally should not be modified.

from dataclasses import dataclass, field
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.attention import sdpa_kernel, SDPBackend
import einops

from pyengine.utils import get_pystratego

pystratego = get_pystratego()

BOARD_LEN = 10
MAX_N_POSSIBLE_DST = 2 * (BOARD_LEN - 1)
N_ACTION = 1800
N_BOARD_CELL = 100


def create_srcdst_to_env_action_index(
    excluded: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create index tensor for converting reduced src-dst action param to environment action param.

    We only need to run this once, so it need not be efficient.

    Args:
        excluded: Tensor of cells excluded from src-dst parameterization.

    Returns:
        (MAX_N_POSSIBLE_DST * N_BOARD_CELL,): Value at index for action in environment param is index for same action in reduced src-dst action param.
            Value -1 means action is not represented in reduced src-dst action param.
    """
    if excluded is None:
        excluded = torch.tensor([], dtype=torch.long)
    n_valid_cells = N_BOARD_CELL - excluded.numel()

    # Mapping from all cells to cells represented by parameterization
    valid_cells = [i for i in range(N_BOARD_CELL) if i not in excluded]
    full_to_reduced = {pos: idx for idx, pos in enumerate(valid_cells)}

    # Tensor of indices to return.
    idx = torch.full((MAX_N_POSSIBLE_DST, N_BOARD_CELL), -1, dtype=torch.long)

    # Iterate over cells for action source.
    for src in range(N_BOARD_CELL):
        # If the action is not reprented in reduced src-dst action param, leave value as -1.
        if src in excluded:
            continue

        # Compute source row and column.
        src_row, src_col = src // BOARD_LEN, src % BOARD_LEN

        # Fill in indices for row moves.
        for new_row in range(BOARD_LEN):
            # If action is not represented in environment param, skip.
            if new_row == src_row:
                continue
            dst = new_row * BOARD_LEN + src_col
            # If action is not represented in reduced src-dst action param, leave value as -1.
            if dst in excluded:
                continue
            movement_idx = (
                new_row if new_row < src_row else new_row - 1
            )  # -1 is to exclude the current cell.
            idx[movement_idx, src] = full_to_reduced[src] * n_valid_cells + full_to_reduced[dst]

        # Fill in indices for column moves.
        for new_col in range(BOARD_LEN):
            # If action is not represented in environment param, skip.
            if new_col == src_col:
                continue
            dst = src_row * BOARD_LEN + new_col
            # If action is not represented in reduced src-dst action param, leave value as -1.
            if dst in excluded:
                continue
            movement_idx = (
                (BOARD_LEN - 1)  # access column movements
                + (new_col if new_col < src_col else new_col - 1)
            )  # -1 is to exclude the current cell.
            idx[movement_idx, src] = full_to_reduced[src] * n_valid_cells + full_to_reduced[dst]

    return idx.view(-1)


def trim_infostate(infostate: torch.Tensor, history_len: int) -> torch.Tensor:
    """Trim the infostate movement planes to `history_len`."""
    board_state = infostate[:, : pystratego.NUM_BOARD_STATE_CHANNELS]
    if history_len == 0:  # Need special case to avoid "-0" indexing
        return board_state
    history_state = infostate[:, pystratego.NUM_BOARD_STATE_CHANNELS :][:, -history_len:]
    return torch.cat([board_state, history_state], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None, is_causal=False, kv_cache=None):
        """
        x: [batch, seq, d_model]
        attn_mask: [seq, seq]
        kv_cache: [2, batch, n_head, lookback_len, d_model]
        """
        qkv = self.qkv_proj(x)
        q, new_k, new_v = einops.rearrange(
            qkv, "b t (k h d) -> b k h t d", k=3, h=self.n_head
        ).unbind(1)
        if kv_cache is not None:
            assert kv_cache.shape[0] == 2
            k = torch.cat([kv_cache[0], new_k], dim=2)
            v = torch.cat([kv_cache[1], new_v], dim=2)
        else:
            k = new_k
            v = new_v
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, attn_mask=attn_mask, is_causal=is_causal
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v), torch.stack([new_k, new_v], dim=0)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout, ff_factor=4):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_head)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_factor * d_model)
        self.linear2 = nn.Linear(ff_factor * d_model, d_model)

        self.dropout = dropout

    def forward(self, x, attn_mask=None, is_causal=False, kv_cache=None):
        x_, new_kv = self.mha(
            self.layer_norm1(x), attn_mask=attn_mask, is_causal=is_causal, kv_cache=kv_cache
        )
        x = x + nn.functional.dropout(x_, p=self.dropout)
        x = x + nn.functional.dropout(self._ff_block(self.layer_norm2(x)), p=self.dropout)
        return x, new_kv

    def _ff_block(self, x):
        x = self.linear2(nn.functional.relu(self.linear1(x)))
        return x


@dataclass
class TransformerRLResumeConfig:
    protect_legacy: Optional[int] = None


@dataclass
class TransformerRLConfig:
    depth: int = 8
    embed_dim_per_head_over8: int = 6
    n_head: int = 8
    dropout: float = 0
    pos_emb_std: float = 0.1
    ff_factor: int = 4
    plane_history_len: int = 32
    use_piece_ids: bool = True
    legacy: bool = False
    protect_legacy: bool = False
    use_threaten: bool = True
    use_evade: bool = True
    use_actadj: bool = True
    use_battle: bool = True
    use_cemetery: bool = True
    use_protect: bool = True
    use_cat_vf: bool = True
    resume: TransformerRLResumeConfig = field(default_factory=lambda: TransformerRLResumeConfig())


class TransformerRL(nn.Module):
    def __init__(self, piece_counts, cfg):
        super().__init__()
        if cfg.legacy:
            self.in_channels = 43 + cfg.plane_history_len
        elif cfg.protect_legacy:
            self.in_channels = 251 + cfg.plane_history_len
        else:
            self.in_channels = pystratego.NUM_BOARD_STATE_CHANNELS + cfg.plane_history_len
        if cfg.use_piece_ids:
            self.in_channels += 2**8

        self.piece_counts = piece_counts
        # Remove emtpy squares and lakes
        # Multiply by 2 because we have two players
        self.total_pieces = 2 * piece_counts[:-2].sum()
        self.cfg = cfg

        # Ensure args satisfy:
        # 1. embed_dim is divisible by n_head
        # 2. embed_dim is divisible by 8
        self.cfg.embed_dim = 8 * cfg.embed_dim_per_head_over8 * cfg.n_head
        self.board_size = 100
        self.lake_indices = torch.tensor([42, 43, 46, 47, 52, 53, 56, 57], dtype=torch.int32)
        action_param_idx = create_srcdst_to_env_action_index(self.lake_indices)
        self.valid_idx = action_param_idx != -1
        self.reduction_idx = action_param_idx[self.valid_idx]
        self.mask = torch.ones(N_BOARD_CELL, dtype=torch.bool)
        self.mask[self.lake_indices] = False
        # Add one extra position for value information
        self.positional_encoding = nn.Parameter(torch.empty(1, self.mask.sum() + 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=cfg.pos_emb_std)
        self.embedder = torch.nn.Linear(self.in_channels, cfg.embed_dim)
        self.norm_out = nn.LayerNorm(cfg.embed_dim)

        self.transformer = nn.Sequential(
            *[
                TransformerLayer(cfg.embed_dim, cfg.n_head, cfg.dropout, cfg.ff_factor)
                for _ in range(cfg.depth)
            ]
        )
        self.qk_proj = nn.Linear(cfg.embed_dim, 2 * cfg.embed_dim)  # (pi) x (q, k)

        if self.cfg.use_cat_vf:
            self.value_head = nn.Linear(cfg.embed_dim, 3)
        else:
            self.value_head = nn.Linear(cfg.embed_dim, 1)

        self.plane_mask = torch.ones(
            pystratego.NUM_BOARD_STATE_CHANNELS + cfg.plane_history_len, dtype=torch.bool
        )
        if not self.cfg.use_threaten:
            self.plane_mask[43:54] = False
            self.plane_mask[76:87] = False
        if not self.cfg.use_evade:
            self.plane_mask[54:65] = False
            self.plane_mask[87:98] = False
        if not self.cfg.use_actadj:
            self.plane_mask[65:76] = False
            self.plane_mask[98:109] = False
        if not self.cfg.use_battle:
            self.plane_mask[131:251] = False
        if not self.cfg.use_cemetery:
            self.plane_mask[109:131] = False
        if not self.cfg.use_protect:
            self.plane_mask[251:355] = False

        self.stats = {}

    def forward(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        legal_action_mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        observations = infostate_tensor
        legal_actions = legal_action_mask
        action_logits, v = self.forward_main(observations, piece_ids, legal_actions)
        dist = Categorical(logits=action_logits)
        actions = dist.sample()
        self.stats["last_action_log_probs"] = dist.log_prob(actions)
        self.stats["last_action_probs"] = dist.log_prob(actions).exp()
        return {
            "action": actions.int(),
            "value": v,
            "action_log_prob": dist.log_prob(actions),
            "action_log_probs": dist.logits.clamp(min=-1e10),
        }

    def forward_main(
        self,
        observations: torch.Tensor,
        piece_ids: torch.Tensor,
        legal_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = trim_infostate(observations, self.cfg.plane_history_len)
        if self.cfg.legacy:
            obs = torch.cat([obs[:, :43], obs[:, -self.cfg.plane_history_len :]], dim=1)
        elif self.cfg.protect_legacy:
            obs = torch.cat([obs[:, :251], obs[:, -self.cfg.plane_history_len :]], dim=1)
        else:
            obs[:, ~self.plane_mask] = 0
        if self.cfg.use_piece_ids:
            one_hot = torch.nn.functional.embedding(
                piece_ids.to(torch.int64), torch.eye(2**8, device=piece_ids.device)
            ).permute(0, 3, 1, 2)
            obs = torch.cat([obs, one_hot], dim=1)
        obs = obs.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        # Remove lake tokens
        obs = obs[:, self.mask]
        # Additional token is for output information
        padded_embedding = F.pad(self.embedder(obs), (0, 0, 1, 0))
        x = padded_embedding + self.positional_encoding
        for layer in self.transformer:
            x, _ = layer(x, attn_mask=None)
        out = self.norm_out(x)
        # Remove any history information
        out = out[:, : self.positional_encoding.shape[1]]
        # 0th embedding is for value information
        v = self.value_head(out[:, 0])
        # Remaining embeddings correspond to cells
        action_logits = self.make_action_logits(out[:, 1:], legal_actions)
        v = v.log_softmax(dim=-1) if self.cfg.use_cat_vf else v.squeeze(-1)
        return action_logits, v

    def make_action_logits(
        self, cell_embeddings: torch.Tensor, legal_actions: torch.Tensor
    ) -> torch.Tensor:
        qk = self.qk_proj(cell_embeddings)
        q, k = einops.rearrange(qk, "b t (k d) -> b k t d", k=2).unbind(1)
        attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        action_logits = torch.zeros(
            attn_weight.shape[0],
            N_ACTION,
            device=attn_weight.device,
            dtype=attn_weight.dtype,
        )
        action_logits[:, self.valid_idx] = attn_weight.flatten(start_dim=1)[:, self.reduction_idx]
        action_logits.masked_fill_(~legal_actions.flatten(start_dim=1).bool(), -float("inf"))
        return action_logits
