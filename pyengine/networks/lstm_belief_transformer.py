from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from pyengine.belief.masking import create_mask
from pyengine.networks.feature_orchestration import FeatureOrchestrator, FeatureOrchestratorConfig
from pyengine.networks.transformer_basics import SelfAttentionLayer, DecoderBlock
from pyengine.utils import get_pystratego
from pyengine.utils.constants import (
    N_OCCUPIABLE_CELL,
    N_PLAYER,
    N_PIECE_TYPE,
    N_CLASSIC_PIECE,
)

pystratego = get_pystratego()


@dataclass
class LSTMBeliefConfig:
    n_encoder_block: int = 6
    n_decoder_block: int = 6
    lstm_before_layer: int = 3
    lstm_residual: bool = True
    lstm_norm: bool = True
    embed_dim_per_head_over8: int = 8
    n_head: int = 8
    dropout: float = 0.2
    pos_emb_std: float = 0.1
    tmp_emb_std: float = 0.1
    ff_factor: int = 4
    only_grounded_features: bool = False
    cheat: bool = False

    @property
    def embed_dim(self):
        return 8 * self.embed_dim_per_head_over8 * self.n_head

    def __post_init__(self):
        assert self.lstm_before_layer < self.n_encoder_block


class LSTMBeliefTransformer(nn.Module):
    """Interleaves spatial attention and temporal attention.

    Spatial attention attends across cells for the board at a particular time step.
    Temporal attention causally attends across time steps for a special token.
    The model is stateless.
    """

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
        self.temporal_layer = nn.LSTM(cfg.embed_dim, cfg.embed_dim)
        self.temporal_layer_norm = nn.LayerNorm(cfg.embed_dim)

        # Our transformer layer abstraction omits the final layer norm, so we need to apply ourselves.
        self.norm_out = nn.LayerNorm(cfg.embed_dim)

        # Decoder
        self.positional_embed_decoder = torch.nn.Linear(N_CLASSIC_PIECE, cfg.embed_dim, bias=False)
        self.piece_embed = nn.Linear(N_PIECE_TYPE, cfg.embed_dim)
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
        cell_embeddings = self.encoder_forward(
            infostate_tensor, piece_ids, num_moves, unknown_piece_position_onehot
        )
        return self.decoder_forward(unknown_piece_type_onehot, cell_embeddings)

    def encoder_forward(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        num_moves: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
    ) -> torch.Tensor:
        x = self.prepare_input(infostate_tensor, piece_ids, num_moves)
        assert torch.all(
            num_moves // N_PLAYER
            == torch.arange(num_moves.shape[-1], device=num_moves.device).unsqueeze(0)
        ), "In sequential mode, num_moves should be [0, 2, 4, ...] for player 0 queries and [1, 3, 5, ...] for player 1 queries"
        x = self.forward_sequential(x)
        return self.extract_mem(x, unknown_piece_position_onehot)

    def decoder_forward(self, unknown_piece_type_onehot, mem):
        shifted_piece_type = F.pad(unknown_piece_type_onehot, (0, 0, 1, 0), value=0)[:, :-1]
        z = self.piece_embed(shifted_piece_type.float())
        z = z + self.positional_embed_decoder.weight.transpose(0, 1).unsqueeze(0)
        for layer in self.decoder:
            z = layer(z, mem)
        logit = self.out_proj(self.decoder_norm_out(z))
        return logit

    def forward_sequential(
        self,
        x: torch.Tensor,  # (batch_size, num_time_steps, num_spatial_plus_special_tokens, embed_dim)
    ) -> torch.Tensor:
        """Assumes first dimension of x indexes time."""
        for i, spatial_layer in enumerate(self.spatial_layers):
            if i == self.cfg.lstm_before_layer:
                x_ = self.temporal_layer_norm(x) if self.cfg.lstm_norm else x
                x_ = self.temporal_layer(x_)[
                    0
                ]  # NOTE: Pytorch LSTM expects (seq, batch, feat) so we don't need to transpose
                x = x + x_ if self.cfg.lstm_residual else x_
            x = spatial_layer(x)
        x = self.norm_out(x)
        return x

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
        x = self.feature_orchestrator(infostate_tensor, piece_ids)
        x = self.embedder(x)
        x += self.positional_encoding.unsqueeze(0)
        x += self.temporal_embedding[: x.shape[0]].unsqueeze(1)
        return x

    def generate_mask(self, seq_len: int, lookback_len: int) -> torch.Tensor:
        """Generates training mask.

        Mask is causal and allows lookback of up to `lookback_len` time steps.
        """
        idx = torch.arange(seq_len)
        delta = idx.unsqueeze(0) - idx.unsqueeze(1)
        return (delta <= 0) & (delta >= -lookback_len)

    def extract_mem(
        self, cell_embeddings: torch.Tensor, unknown_piece_position_onehot
    ) -> torch.Tensor:
        piece_pos = unknown_piece_position_onehot[..., self.feature_orchestrator.cell_mask]
        hidden_embeddings = torch.gather(
            cell_embeddings,
            1,
            piece_pos.int().argmax(dim=-1, keepdim=True).expand(-1, -1, cell_embeddings.size(-1)),
        )
        is_active = piece_pos.any(dim=-1, keepdim=True)
        hidden_embeddings.masked_fill_(~is_active, 0)
        return hidden_embeddings

    def sample(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        num_moves: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        n_sample: int,
    ):
        assert infostate_tensor.ndim == 4
        assert piece_ids.ndim == 3
        assert num_moves.ndim == 1
        assert unknown_piece_position_onehot.ndim == 3
        assert unknown_piece_counts.ndim == 2
        assert unknown_piece_has_moved.ndim == 2
        # Make memory
        mem = self.encoder_forward(
            infostate_tensor, piece_ids, num_moves, unknown_piece_position_onehot
        )[-1].unsqueeze(0)
        # Reshape for sampling
        samples = torch.zeros(
            n_sample,
            unknown_piece_has_moved.shape[-1],
            pystratego.NUM_PIECE_TYPES,
            dtype=torch.bool,
            device="cuda",
        )
        mem = mem.repeat(n_sample, 1, 1)
        unknown_piece_counts = unknown_piece_counts[-1].repeat(n_sample, 1)
        unknown_piece_has_moved = unknown_piece_has_moved[-1].repeat(n_sample, 1)
        unknown_piece_position_onehot = unknown_piece_position_onehot[-1]
        # Sample
        samples_idx = torch.arange(samples.shape[0], device="cuda")
        needs_sample = unknown_piece_position_onehot.any(dim=-1)
        for i in range(unknown_piece_has_moved.shape[1]):
            if not needs_sample[i]:
                break
            logits = self.decoder_forward(samples, mem)
            mask = create_mask(unknown_piece_counts, samples, unknown_piece_has_moved)
            assert mask.any(dim=-1).all()
            logits.masked_fill_(~mask, -1e10)
            conditional = Categorical(logits=logits[:, i])
            pieces = conditional.sample()
            samples[samples_idx, i, pieces] = True
        return samples
