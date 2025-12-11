from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from pyengine.networks.feature_orchestration import FeatureOrchestrator, FeatureOrchestratorConfig
from pyengine.networks.transformer_basics import SelfAttentionLayer
from pyengine.networks.utils import LogitConverter
from pyengine.utils import get_pystratego
from pyengine.utils.constants import N_OCCUPIABLE_CELL, N_VF_CAT

pystratego = get_pystratego()


@dataclass
class MoveTransformerConfig:
    depth: int = 8
    embed_dim_per_head_over8: int = 6
    n_head: int = 8
    dropout: float = 0
    pos_emb_std: float = 0.1
    ff_factor: int = 4
    use_cat_vf: bool = True
    plane_history_len: int = 32
    use_planes: bool = True

    @property
    def embed_dim(self):
        return 8 * self.embed_dim_per_head_over8 * self.n_head


class MoveTransformer(nn.Module):
    def __init__(self, piece_counts, cfg):
        super().__init__()
        self.piece_counts = piece_counts
        self.cfg = cfg

        if cfg.use_planes:
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
        else:
            self.feature_orchestrator = FeatureOrchestrator(
                FeatureOrchestratorConfig(
                    use_piece_ids=True,
                    use_threaten=False,
                    use_evade=False,
                    use_actadj=False,
                    use_cemetery=True,
                    use_battle=False,
                    use_protect=False,
                    plane_history_len=cfg.plane_history_len,
                )
            )
        self.special_token_indices = {"value": 0}
        self.positional_encoding = nn.Parameter(
            torch.empty(1, N_OCCUPIABLE_CELL + len(self.special_token_indices), cfg.embed_dim)
        )
        nn.init.trunc_normal_(self.positional_encoding, std=cfg.pos_emb_std)
        self.embedder = torch.nn.Linear(self.feature_orchestrator.in_channels, cfg.embed_dim)
        self.norm_out = nn.LayerNorm(cfg.embed_dim)

        self.transformer = nn.Sequential(
            *[
                SelfAttentionLayer(cfg.embed_dim, cfg.n_head, cfg.dropout, cfg.ff_factor)
                for _ in range(cfg.depth)
            ]
        )
        self.q_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.k_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.logit_converter = LogitConverter()

        if self.cfg.use_cat_vf:
            self.value_head = nn.Linear(cfg.embed_dim, N_VF_CAT)
        else:
            self.value_head = nn.Linear(cfg.embed_dim, 1)

        self.stats = {}

    def forward(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        legal_action_mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        action_logits, v = self.forward_main(
            infostate_tensor,
            piece_ids,
            legal_action_mask,
        )
        dist = Categorical(logits=action_logits)
        actions = dist.sample()
        self.stats["last_action_log_probs"] = dist.log_prob(actions)
        self.stats["last_action_probs"] = dist.log_prob(actions).exp()
        return {
            "action": actions.int(),
            "value": v,
            "action_log_prob": dist.log_prob(actions),
            "action_log_probs": dist.logits.log_softmax(dim=-1),
        }

    def forward_main(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        legal_action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_orchestrator(infostate_tensor, piece_ids)
        x = self.embedder(x)
        x = F.pad(x, (0, 0, len(self.special_token_indices), 0))
        x = x + self.positional_encoding
        for layer in self.transformer:
            x = layer(x)
        out = self.norm_out(x)
        action_logits = self.make_action_logits(
            out[..., len(self.special_token_indices) :, :], legal_action_mask
        )
        v = self.value_head(out[..., self.special_token_indices["value"], :])
        v = v.log_softmax(dim=-1) if self.cfg.use_cat_vf else v.squeeze(-1)
        return action_logits, v

    def make_action_logits(
        self, cell_embeddings: torch.Tensor, legal_action_mask: torch.Tensor
    ) -> torch.Tensor:
        q = self.q_proj(cell_embeddings)
        k = self.k_proj(cell_embeddings)
        attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        action_logits = self.logit_converter(attn_weight)
        action_logits.masked_fill_(~legal_action_mask, torch.finfo(action_logits.dtype).min)
        return action_logits
