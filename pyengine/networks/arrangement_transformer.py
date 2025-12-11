from dataclasses import dataclass
import torch
import torch.nn as nn

from pyengine.utils.validation import expect_shape, expect_device
from pyengine.networks.transformer_basics import SelfAttentionLayer
from pyengine.utils.constants import (
    ARRANGEMENT_SIZE,
    FLAG_IDX,
    N_ARRANGEMENT_ROW,
    N_ARRANGEMENT_COL,
    N_PIECE_TYPE,
    N_VF_CAT,
)


@dataclass
class ArrangementTransformerConfig:
    depth: int = 4
    embed_dim_per_head_over8: int = 8
    n_head: int = 8
    pos_emb_std: float = 0.1
    use_cat_vf: bool = True
    force_handedness: bool = True

    @property
    def embed_dim(self):
        return 8 * self.embed_dim_per_head_over8 * self.n_head


class ArrangementTransformer(nn.Module):
    """Network for generating arrangements.

    The network is designed to generate arrangements autoregressively by placing pieces from bottom to top, left to right.
    It internally handles masking out invalid piece placements, so placements with positive probability are guaranteed to be valid.
    Since equilibrium policies are left-right symmetric, the network supports forcing right-handedness (i.e. right-side flags)
    to eliminate an axis of symmetry.
    NOTE: If `force_handedness` is True, the arrangement handedness should be randomized post-network generation.
    Otherwise, the flag will always be on the right side.
    """

    def __init__(
        self,
        piece_counts: torch.Tensor,
        cfg: ArrangementTransformerConfig,
    ) -> None:
        super().__init__()
        if not isinstance(piece_counts, torch.Tensor):
            raise TypeError("piece_counts must be a tensor")
        if piece_counts.size(0) != N_PIECE_TYPE:
            raise ValueError(
                f"piece_counts must have {N_PIECE_TYPE} elements, got {piece_counts.size(0)}"
            )
        if piece_counts.sum() != ARRANGEMENT_SIZE:
            raise ValueError(
                f"piece_counts must sum to {ARRANGEMENT_SIZE}, got {piece_counts.sum()}"
            )
        self.cfg = cfg

        self.embedder = torch.nn.Linear(N_PIECE_TYPE, cfg.embed_dim)
        self.positional_encoding = nn.Parameter(torch.empty(1, ARRANGEMENT_SIZE, cfg.embed_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=cfg.pos_emb_std)
        self.register_buffer("start_token", torch.zeros(1, 1, N_PIECE_TYPE))

        self.transformer = nn.ModuleList(
            [SelfAttentionLayer(cfg.embed_dim, cfg.n_head, dropout=0) for _ in range(cfg.depth)]
        )
        self.norm_out = nn.LayerNorm(cfg.embed_dim)

        self.register_buffer("piece_counts", piece_counts)
        if cfg.force_handedness:
            self.register_buffer(
                "right_side",
                torch.tensor(
                    N_ARRANGEMENT_ROW
                    * (N_ARRANGEMENT_COL // 2 * [False] + N_ARRANGEMENT_COL // 2 * [True]),
                    dtype=torch.bool,
                ),
            )
        else:
            self.right_side = None
        self.policy_out = nn.Linear(cfg.embed_dim, N_PIECE_TYPE)
        if self.cfg.use_cat_vf:
            self.value_out = nn.Linear(cfg.embed_dim, N_VF_CAT)
        else:
            self.value_out = nn.Linear(cfg.embed_dim, 1)
        self.ent_out = nn.Linear(cfg.embed_dim, 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict logits, value, and future accumulated entropy for each prefix of each arrangement.

        Args:
            seq (B, <=ARRANGEMENT_SIZE, N_PIECE_TYPE): Batch of arrangement prefixes.
                First row is the first piece (bottom left), second row is second piece (bottom second-from-left), and so forth.

        Returns:
            (B, <=ARRANGEMENT_SIZE, N_PIECE_TYPE): Logits for each piece type.
            (B, <=ARRANGEMENT_SIZE, 1) OR (B, <=ARRANGEMENT_SIZE, N_VF_CAT): Value prediction.
            (B, <=ARRANGEMENT_SIZE, 1): Future accumulated entropy prediction.
        """
        self._check_seq(seq)

        start_token = self.start_token.expand(seq.shape[0], -1, -1)
        seq = torch.cat([start_token, seq], dim=1)[:, :ARRANGEMENT_SIZE]

        seq_embedding = self.embedder(seq)
        x = seq_embedding + self.positional_encoding[:, : seq_embedding.shape[1]]

        for layer in self.transformer:
            x = layer(x, is_causal=True)
        x = self.norm_out(x)

        logits = self.policy_out(x)
        value = self.value_out(x)
        ent_pred = self.ent_out(x)

        legal_action_mask = self._create_legal_action_mask(seq)
        logits.masked_fill_(~legal_action_mask, torch.finfo(logits.dtype).min)

        return {"logits": logits, "value": value, "ent_pred": ent_pred}

    @torch.no_grad()
    def _create_legal_action_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create mask of legal actions.

        Args:
            seq (B, <=ARRANGEMENT_SIZE, N_PIECE_TYPE): Arrangement prefix.

        Returns:
            (B, ARRANGEMENT_SIZE, N_PIECE_TYPE): Mask of legal actions.
        """
        assert (seq[:, 0] == self.start_token).all(), "seq must start with start token"
        legal_mask = self._remaining_counts(seq) > 0
        if self.right_side is not None:
            legal_mask[:, ~self.right_side[: legal_mask.size(1)], FLAG_IDX] = False
        assert legal_mask.any(dim=-1).all(), "seq must have at least one legal action"
        return legal_mask

    @torch.no_grad()
    def _remaining_counts(self, seq: torch.Tensor) -> torch.Tensor:
        """Count the number of remaining pieces to be placed for each piece type.

        Args:
            seq (B, <=ARRANGEMENT_SIZE, N_PIECE_TYPE): Arrangement prefix.

        Returns:
            (B, <=ARRANGEMENT_SIZE, N_PIECE_TYPE): Number of remaining pieces to be placed for each piece type.
        """
        cum_counts = seq.cumsum(dim=1)
        expanded_piece_counts = self.piece_counts.view(1, 1, -1).expand_as(cum_counts)
        remaining = expanded_piece_counts - cum_counts
        assert (
            remaining[:, 0] == self.piece_counts
        ).all(), "initial remaining counts must match piece counts"
        assert (remaining > 0).any(dim=-1).all(), "remaining counts must be positive"
        return remaining

    def _check_seq(self, seq: torch.Tensor) -> None:
        """Check that seq is a valid arrangement prefix."""
        expect_shape(seq, ndim=3, dims={2: N_PIECE_TYPE}, name="seq")
        expect_device(seq, self.device, name="seq")
        if not (seq.shape[1] <= ARRANGEMENT_SIZE):
            raise ValueError(f"seq must have at most {ARRANGEMENT_SIZE} rows, got {seq.shape[1]}.")
