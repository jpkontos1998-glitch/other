from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from pyengine.utils.types import GenerateArgType
from pyengine.utils.validation import expect_shape, expect_same_batch
from pyengine.networks.feature_orchestration import FeatureOrchestrator, FeatureOrchestratorConfig
from pyengine.networks.transformer_basics import SelfAttentionLayer, DecoderBlock
from pyengine.networks.utils import extract_mem
from pyengine.belief.sampling import sampling_loop, check_sampling_args
from pyengine.utils import get_pystratego
from pyengine.utils.validation import expect_type
from pyengine.utils.constants import (
    N_OCCUPIABLE_CELL,
    N_PLAYER,
    N_PIECE_TYPE,
    N_BARRAGE_PIECE,
    N_CLASSIC_PIECE,
    BOARD_LEN,
)

pystratego = get_pystratego()


@dataclass
class TemporalBeliefConfig:
    n_encoder_block: int = 6
    n_decoder_block: int = 6
    embed_dim_per_head_over8: int = 8
    n_head: int = 8
    dropout: float = 0.2
    pos_emb_std: float = 0.1
    tmp_emb_std: float = 0.1
    ff_factor: int = 4
    only_grounded_features: bool = False
    barrage: bool = False

    @property
    def embed_dim(self):
        return 8 * self.embed_dim_per_head_over8 * self.n_head


class TemporalBeliefTransformer(nn.Module):
    """Interleaves spatial attention and temporal attention.

    Spatial attention attends across cells for the board at a particular time step.
    Temporal attention causally attends across time steps for a special token.
    The model is stateless.
    """

    generate_arg_type = GenerateArgType.TEMPORAL_TRANSFORMER

    def __init__(self, max_num_moves, cfg):
        super().__init__()
        self.max_num_moves = max_num_moves
        self.cfg = cfg

        if cfg.only_grounded_features:
            self.feature_orchestrator = FeatureOrchestrator(
                FeatureOrchestratorConfig(
                    use_piece_ids=True,
                    use_threaten=False,
                    use_evade=False,
                    use_actadj=False,
                    use_cemetery=True,
                    use_battle=False,
                    use_protect=False,
                    plane_history_len=0,
                )
            )
        else:
            self.feature_orchestrator = FeatureOrchestrator(
                FeatureOrchestratorConfig(
                    use_piece_ids=True,
                    use_threaten=True,
                    use_evade=True,
                    use_actadj=True,
                    use_cemetery=True,
                    use_battle=True,
                    use_protect=True,
                    plane_history_len=32,
                )
            )

        self.embedder = torch.nn.Linear(self.feature_orchestrator.in_channels, cfg.embed_dim)
        # Positional embeddings for both positions and time.
        self.positional_encoding = nn.Parameter(torch.empty(N_OCCUPIABLE_CELL, cfg.embed_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=cfg.pos_emb_std)
        self.temporal_embedding = nn.Parameter(
            torch.empty(max_num_moves // N_PLAYER + 2, cfg.embed_dim)  # +2 for dummy terminal moves
        )
        nn.init.trunc_normal_(self.temporal_embedding, std=cfg.tmp_emb_std)

        # The spatial layers attend over (board cells \cup special tokens) for a particular time step.
        self.spatial_layers = nn.ModuleList(
            [
                SelfAttentionLayer(cfg.embed_dim, cfg.n_head, cfg.dropout, cfg.ff_factor)
                for _ in range(cfg.n_encoder_block)
            ]
        )
        # The temporal layers attend over temporal tokens for different time steps.
        self.temporal_layers = nn.ModuleList(
            [
                SelfAttentionLayer(cfg.embed_dim, cfg.n_head, cfg.dropout, cfg.ff_factor)
                for _ in range(cfg.n_encoder_block)
            ]
        )

        # Our transformer layer abstraction omits the final layer norm, so we need to apply ourselves.
        self.norm_out = nn.LayerNorm(cfg.embed_dim)

        # Decoder
        if cfg.barrage:
            n_piece = N_BARRAGE_PIECE
        else:
            n_piece = N_CLASSIC_PIECE
        self.positional_embed_dec = nn.Parameter(torch.empty(n_piece, cfg.embed_dim))
        nn.init.trunc_normal_(self.positional_embed_dec, std=cfg.pos_emb_std)
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(cfg.embed_dim, cfg.n_head, cfg.dropout)
                for _ in range(cfg.n_decoder_block)
            ]
        )
        self.decoder_norm_out = nn.LayerNorm(cfg.embed_dim)
        self.out_proj = nn.Linear(cfg.embed_dim, N_PIECE_TYPE)

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
        if not torch.all(
            num_moves // N_PLAYER
            == torch.arange(num_moves.shape[-1], device=num_moves.device).unsqueeze(0)
        ):
            raise ValueError(
                "num_moves should be [0, 2, 4, ...] for player 0 queries and [1, 3, 5, ...] for player 1 queries"
            )
        mem = self.encoder_forward(
            infostate_tensor, piece_ids, num_moves, unknown_piece_position_onehot
        )
        return self.decoder_forward(unknown_piece_type_onehot, mem)

    def encoder_forward(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        num_moves: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
    ) -> torch.Tensor:
        x = self.prepare_input(infostate_tensor, piece_ids, num_moves)
        for spatial_layer, temporal_layer in zip(self.spatial_layers, self.temporal_layers):
            x = spatial_layer(x)
            x = temporal_layer(x.transpose(0, 1), is_causal=True).transpose(0, 1)
        mem = extract_mem(x, unknown_piece_position_onehot, self.feature_orchestrator.cell_mask)
        return self.norm_out(mem)

    def decoder_forward(self, unknown_piece_type_onehot, mem):
        shifted_piece_type = F.pad(unknown_piece_type_onehot, (0, 0, 1, 0))[:, :-1]
        x = self.piece_embed(shifted_piece_type.float())
        z = x + self.positional_embed_dec.unsqueeze(0)
        for layer in self.decoder:
            z = layer(z, mem)
        logit = self.out_proj(self.decoder_norm_out(z))
        return logit

    def forward_sequential(
        self,
        x: torch.Tensor,  # (batch_size, num_time_steps, num_spatial_plus_special_tokens, embed_dim)
    ) -> torch.Tensor:
        """Assumes first dimension of x indexes time."""

    def prepare_input(
        self, infostate_tensor: torch.Tensor, piece_ids: torch.Tensor, num_moves: torch.Tensor
    ):
        """Aggregate input information into single tensor.

        This involves:
        - Tasks handled by the feature orchestrator:
            - Removing unused planes from the infostate.
            - Translating the piece IDs into one-hot encodings and concatenating them to the infostate.
            - Removing the lake tokens from the infostate.
        - Adding padding for the special tokens.
        - Adding positional embeddings.
        - Adding temporal embeddings.
        """
        x = self.embedder(self.feature_orchestrator(infostate_tensor, piece_ids))
        x += self.positional_encoding.unsqueeze(0)
        x += self.temporal_embedding[: x.shape[0]].unsqueeze(1)
        return x

    def generate(
        self,
        n_sample: int,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        num_moves: torch.Tensor,
    ):
        expect_type(n_sample, int, name="n_sample")
        check_sampling_args(
            unknown_piece_position_onehot, unknown_piece_has_moved, unknown_piece_counts
        )
        expect_shape(
            infostate_tensor, ndim=4, dims={2: BOARD_LEN, 3: BOARD_LEN}, name="infostate_tensor"
        )
        expect_shape(piece_ids, ndim=3, dims={1: BOARD_LEN, 2: BOARD_LEN}, name="piece_ids")
        expect_shape(num_moves, ndim=1, name="num_moves")
        expect_same_batch(infostate_tensor, piece_ids, num_moves)
        # Make memory
        n_step = infostate_tensor.shape[0]
        mem = (
            self.encoder_forward(
                infostate_tensor,
                piece_ids,
                num_moves,
                unknown_piece_position_onehot.unsqueeze(0).expand(n_step, -1, -1),
            )[-1]
            .unsqueeze(0)
            .expand(n_sample, -1, -1)
        )

        def partial(samples: torch.Tensor) -> torch.Tensor:
            return self.decoder_forward(samples, mem)

        return sampling_loop(
            n_sample,
            unknown_piece_position_onehot,
            unknown_piece_has_moved,
            unknown_piece_counts,
            partial,
        )
