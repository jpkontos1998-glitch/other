from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch import nn
import torch.distributed as dist

from pyengine.arrangement.sampling import generate_arrangements
from pyengine.arrangement.utils import to_string, filter_terminal
from pyengine.core.env import Stratego
from pyengine.belief.buffer import CircularBuffer, Batch
from pyengine.belief.uniform import marginalized_uniform_belief, uniform_belief
from pyengine.networks.lstm_belief_transformer import LSTMBeliefConfig
from pyengine.networks.belief_transformer import BeliefTransformerConfig
from pyengine.networks.temporal_belief_transformer import TemporalBeliefConfig
from pyengine.networks.exponential_weighted_average import EMA
from pyengine import utils

pystratego = utils.get_pystratego()


@dataclass
class BeliefConfig:
    from_checkpoint: Optional[str] = None  # leave None to train from scratch
    # belief params
    belief_transformer: BeliefTransformerConfig = field(default_factory=BeliefTransformerConfig)
    temporal_belief: TemporalBeliefConfig = field(default_factory=TemporalBeliefConfig)
    lstm_belief: LSTMBeliefConfig = field(default_factory=LSTMBeliefConfig)
    use_lstm_model: bool = False
    use_temporal_model: bool = False
    # optim
    lr: float = 5e-5
    max_grad_norm: float = 0.5
    num_batch_per_train: int = 100
    max_num_train: int = int(1e9)
    # others
    save_every: int = 1000
    dtype: str = "bfloat16"
    n_arr: int = 1024
    # ema
    ema_decay: float = 0.99
    # new params
    num_row: int = 200
    torch_compile: bool = True

    def get_dtype(self):
        return {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.dtype]

    def __post_init__(self):
        assert not (self.use_lstm_model and self.use_temporal_model)


class Belief:
    def __init__(
        self,
        belief_model: nn.Module,
        rl_model: nn.Module,
        arr_model: nn.Module,
        env: Stratego,
        stats: utils.MultiCounter,
        saver: utils.Saver,
        cfg: BeliefConfig,
        rank: int = 0,
        saver_enabled: bool = True,
    ):
        self.belief_model = belief_model
        self.rl_model = rl_model
        self.arr_model = arr_model
        self.env = env
        self.cfg = cfg
        self.dtype = cfg.get_dtype()
        self.rank = rank
        self.saver_enabled = saver_enabled

        self.optim = torch.optim.Adam(self.belief_model.parameters(), lr=self.cfg.lr)

        if cfg.torch_compile:
            self.belief_model = torch.compile(self.belief_model)
            self.rl_model = torch.compile(self.rl_model)

        if cfg.from_checkpoint is not None:
            optim_path = cfg.from_checkpoint.replace(".pthw", ".ptho")
            print(f"Loading optimizer state from: {optim_path}")
            self.optim.load_state_dict(torch.load(optim_path))
            print(f"Loading weights from: {cfg.from_checkpoint}")
            belief_model.load_state_dict(torch.load(cfg.from_checkpoint))

        self.ema = EMA(belief_model, cfg.ema_decay)
        if cfg.from_checkpoint is not None:
            ema_path = cfg.from_checkpoint.replace(".pthw", ".pthm")
            print(f"Loading ema model from: {ema_path}")
            self.ema.ema_model.load_state_dict(torch.load(ema_path))

        self.buffer = CircularBuffer(
            num_envs=env.num_envs,
            num_row=cfg.num_row,
            max_num_moves=env.conf.max_num_moves,
            max_num_moves_between_attacks=env.conf.max_num_moves_between_attacks,
            device=self.rank,
        )

        # track progress
        self.num_env_step = 0
        self.num_train_step = 0
        self.num_train_batch = 0
        self.num_log_step = 0
        self.stats = stats
        self.saver = saver
        self.stopwatch = utils.Stopwatch()

    def learn(self):
        self.num_train_step = 0
        while self.num_train_step < self.cfg.max_num_train:
            with self.stopwatch.time("update_arrs"):
                with torch.no_grad(), utils.eval_mode(self.arr_model):
                    self.update_arrs()

            with self.stopwatch.time("rollout"):
                with torch.no_grad(), utils.eval_mode(self.rl_model):
                    self.collect_rollouts()

            with self.stopwatch.time("train"):
                self.train()

            self.num_train_step += 1

            self.stats["count/num_train_step"].append(self.num_train_step)
            self.stats["count/num_train_batch"].append(self.num_train_batch)
            self.stats[f"count/num_env_step({self.env.num_envs}s)"].append(self.num_env_step)

            for key, val in self.env.stats.items():
                self.stats[f"data/{key}"].append(val)

            time_data = self.stopwatch.summary()
            for k, v in time_data.items():
                self.stats[k].append(v)
            self.stats.summary(self.num_train_step, reset=True)
            self.env.reset_stats()

            if self.saver_enabled:
                if self.num_train_step % self.cfg.save_every == 0 and self.num_train_step > 0:
                    self.saver.save(
                        self.belief_model.state_dict(),
                        self.ema.ema_model.state_dict(),
                        self.optim.state_dict(),
                        model_id=self.num_train_step,
                    )

            self.num_log_step += 1

            # Eval, save, terminate if we are running out of time
            time_left = utils.get_slurm_remaining_time_minutes()
            if time_left is not None and time_left < 10:
                print("Initiating terminal save (reason: time limit)")
                if self.saver_enabled:
                    self.saver.save(
                        self.belief_model.state_dict(),
                        self.ema.ema_model.state_dict(),
                        self.optim.state_dict(),
                        model_id=self.num_train_step,
                    )

                self.stopwatch.summary()
                self.stats.summary(self.num_train_step, reset=True)
                print("JOB COMPLETED SUCCESSFULLY")
                return

    def train(self) -> None:
        assert self.belief_model.training
        buffer = self.buffer
        env = self.env

        n_batch = buffer.is_newly_terminal.sum()
        if dist.is_initialized():
            dist.all_reduce(n_batch, op=dist.ReduceOp.SUM)
        n_batch = n_batch.item()
        if n_batch == 0:
            return

        n_batch = n_batch // self.cfg.num_batch_per_train
        for i, batch in enumerate(buffer.sample(env)):
            if i == n_batch:
                break

            ongoing = torch.tensor([batch is not None], device=self.rank, dtype=torch.bool)
            if dist.is_initialized():
                dist.all_reduce(ongoing, op=dist.ReduceOp.MIN)
            if not ongoing.item():
                break

            # Optimization step
            with autocast(device_type="cuda", dtype=self.dtype):
                loss = self.compute_and_record_loss(batch)

            self.optim.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(  # type: ignore
                self.belief_model.parameters(), self.cfg.max_grad_norm
            )
            self.optim.step()

            self.num_train_batch += 1
            self.stats["train/g_norm"].append(grad_norm)
            self.stats["train/lr"].append(self.optim.param_groups[0]["lr"])
        self.ema.update()
        buffer.reset()

    def compute_and_record_loss(self, batch: Batch):
        infostate = batch.infostate_tensor
        piece_ids = batch.piece_ids
        num_moves = batch.num_moves
        unknown_piece_position_onehot = batch.unknown_piece_position_onehot
        unknown_piece_type_onehot = batch.unknown_piece_type_onehot
        unknown_piece_counts = batch.unknown_piece_counts
        unknown_piece_has_moved = batch.unknown_piece_has_moved

        logits = self.belief_model(
            infostate_tensor=infostate,
            piece_ids=piece_ids,
            num_moves=num_moves,
            unknown_piece_position_onehot=unknown_piece_position_onehot,
            unknown_piece_type_onehot=unknown_piece_type_onehot,
            unknown_piece_counts=unknown_piece_counts,
            unknown_piece_has_moved=unknown_piece_has_moved,
        )
        log_prob = F.log_softmax(logits, dim=-1)
        active_pieces = unknown_piece_type_onehot.any(-1)

        ce_loss = cross_entropy(active_pieces, unknown_piece_type_onehot, log_prob).mean()

        mu_belief = marginalized_uniform_belief(infostate, unknown_piece_position_onehot)
        mu_ce = cross_entropy(active_pieces, mu_belief.exp(), log_prob).mean()
        mu_ent = cross_entropy(active_pieces, mu_belief.exp(), mu_belief).mean()
        mu_kl = mu_ce - mu_ent

        u_belief = uniform_belief(
            unknown_piece_type_onehot,
            unknown_piece_has_moved,
            unknown_piece_counts,
        )
        u_ce = cross_entropy(active_pieces, u_belief.exp(), log_prob).mean()
        u_ent = cross_entropy(active_pieces, u_belief.exp(), u_belief).mean()
        u_kl = u_ce - u_ent

        self.stats["train/cross_entropy"].append(ce_loss.item())
        self.stats["train/marginalized_uniform_kl"].append(mu_kl.item())
        self.stats["train/uniform_kl"].append(u_kl.item())

        return ce_loss

    def collect_rollouts(self):
        assert not self.rl_model.training
        while not self.buffer.ready_to_train():
            self.buffer.add(
                obs_step=self.env.current_step,
                is_newly_terminal=self.env.current_is_newly_terminal,
            )
            with torch.no_grad(), utils.eval_mode(self.rl_model), autocast(
                device_type="cuda", dtype=self.dtype
            ):
                tensor_dict = self.rl_model(
                    self.env.current_infostate_tensor,
                    self.env.current_piece_ids,
                    self.env.current_legal_action_mask,
                )
            self.env.apply_actions(tensor_dict["action"])

    def update_arrs(self) -> None:
        with torch.no_grad(), utils.eval_mode(self.arr_model):
            arrs, *_ = generate_arrangements(self.cfg.n_arr, self.arr_model)
        arrs = filter_terminal(to_string(arrs))
        self.env.change_reset_behavior_to_random_initial_arrangement((arrs[::2], arrs[1::2]))
        if self.num_train_step == 0:
            self.env.reset()


def cross_entropy(active_pieces: torch.Tensor, prob: torch.Tensor, log_prob: torch.Tensor):
    """
    Args:
        active_pieces (B, n_piece): Whether the piece is active
        prob (B, n_piece, N_PIECE_TYPE): Probability of each piece type
        log_prob (B, n_piece, N_PIECE_TYPE): Log probability of each piece type

    Returns:
        (B,)
    """
    xent = -(prob * log_prob).sum(-1)
    xent = (xent * active_pieces).sum(-1) / active_pieces.sum(-1)
    return xent
