from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from pyengine.utils.validation import expect_shape, expect_type
from pyengine.utils.types import GenerateArgType
from pyengine.networks.feature_orchestration import FeatureOrchestrator, FeatureOrchestratorConfig
from pyengine.belief.sampling import sampling_loop, check_sampling_args
from pyengine.networks.transformer_basics import SelfAttentionLayer, DecoderBlock
from pyengine.networks.utils import extract_mem
from pyengine.utils import get_pystratego
from pyengine.utils.constants import (
    N_PIECE_TYPE,
    N_OCCUPIABLE_CELL,
    BOARD_LEN,
    N_BARRAGE_PIECE,
    N_CLASSIC_PIECE,
)

pystratego = get_pystratego()


@dataclass
class BeliefTransformerConfig:
    n_encoder_layer: int = 6
    n_decoder_block: int = 6
    num_head: int = 8
    embed_dim_per_head_over8: int = 8
    dropout: float = 0.2
    pos_emb_std: float = 0.1
    plane_history_len: int = 86
    barrage: bool = False

    @property
    def embed_dim(self) -> int:
        return self.embed_dim_per_head_over8 * self.num_head * 8


class BeliefTransformer(nn.Module):
    generate_arg_type = GenerateArgType.PLANAR_TRANSFORMER

    def __init__(self, cfg: BeliefTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.feature_orchestrator = FeatureOrchestrator(
            FeatureOrchestratorConfig(
                use_piece_ids=True,
                use_threaten=True,
                use_evade=True,
                use_actadj=True,
                use_cemetery=True,
                use_battle=True,
                use_protect=True,
                plane_history_len=cfg.plane_history_len,
            )
        )

        self.embedder = torch.nn.Linear(self.feature_orchestrator.in_channels, cfg.embed_dim)
        self.positional_embed_enc = nn.Parameter(torch.empty(N_OCCUPIABLE_CELL, cfg.embed_dim))
        nn.init.trunc_normal_(self.positional_embed_enc, std=cfg.pos_emb_std)
        self.piece_embed = nn.Linear(N_PIECE_TYPE, cfg.embed_dim)
        if cfg.barrage:
            n_piece = N_BARRAGE_PIECE
        else:
            n_piece = N_CLASSIC_PIECE
        self.positional_embed_dec = nn.Parameter(torch.empty(n_piece, cfg.embed_dim))
        nn.init.trunc_normal_(self.positional_embed_dec, std=cfg.pos_emb_std)

        self.encoder = nn.ModuleList(
            [
                SelfAttentionLayer(cfg.embed_dim, cfg.num_head, cfg.dropout)
                for _ in range(cfg.n_encoder_layer)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(cfg.embed_dim, cfg.num_head, cfg.dropout)
                for _ in range(cfg.n_decoder_block)
            ]
        )
        self.final_ln_enc = nn.LayerNorm(cfg.embed_dim)
        self.final_ln_dec = nn.LayerNorm(cfg.embed_dim)
        self.final_linear = nn.Linear(cfg.embed_dim, N_PIECE_TYPE)

    def forward(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_type_onehot: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        mem = self.encoder_forward(infostate_tensor, piece_ids, unknown_piece_position_onehot)
        return self.decoder_forward(unknown_piece_type_onehot, mem)

    def encoder_forward(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedder(self.feature_orchestrator(infostate_tensor, piece_ids))
        x = x + self.positional_embed_enc.unsqueeze(0)
        for layer in self.encoder:
            x = layer(x)
        mem = extract_mem(x, unknown_piece_position_onehot, self.feature_orchestrator.cell_mask)
        return self.final_ln_enc(mem)

    def decoder_forward(self, unknown_piece_type_onehot, mem):
        shifted_piece_type = F.pad(unknown_piece_type_onehot, (0, 0, 1, 0))[:, :-1]
        x = self.piece_embed(shifted_piece_type.float())
        x = x + self.positional_embed_dec.unsqueeze(0)
        for layer in self.decoder:
            x = layer(x, mem)
        logit = self.final_linear(self.final_ln_dec(x))

        return logit

    def generate(
        self,
        n_sample: int,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
    ) -> torch.Tensor:
        expect_type(n_sample, "n_sample", int)
        check_sampling_args(
            unknown_piece_position_onehot, unknown_piece_has_moved, unknown_piece_counts
        )
        expect_shape(infostate_tensor, ndim=3, dims={1: BOARD_LEN, 2: BOARD_LEN})
        expect_shape(piece_ids, ndim=2, dims={0: BOARD_LEN, 1: BOARD_LEN})
        mem = self.encoder_forward(
            infostate_tensor.unsqueeze(0), unknown_piece_position_onehot.unsqueeze(0)
        ).expand(n_sample, -1, -1)

        def partial(samples: torch.Tensor) -> torch.Tensor:
            return self.decoder_forward(samples, mem)

        return sampling_loop(
            n_sample,
            unknown_piece_position_onehot,
            unknown_piece_has_moved,
            unknown_piece_counts,
            partial,
        )
