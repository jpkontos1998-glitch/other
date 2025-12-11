from dataclasses import dataclass, field
import signal
from typing import Optional
import os
import math

import torch
import torch.nn.functional as F
from torch.amp import autocast
import torch.distributed as dist
import pickle

from pyengine.arrangement.buffer import ArrangementBuffer
from pyengine.arrangement.sampling import generate_arrangements
from pyengine.arrangement.utils import to_string, filter_terminal
from pyengine.core.env import Stratego
from pyengine.core.evaluation import EvaluationManager
from pyengine.core.buffer import CircularBuffer
from pyengine.core.train_container import TrainContainer
from pyengine.networks.arrangement_transformer import (
    ArrangementTransformer,
    ArrangementTransformerConfig,
)
from pyengine.networks.move_transformer import MoveTransformer, MoveTransformerConfig
from pyengine.networks.exponential_weighted_average import EMA
from pyengine.utils.distributed_checks import check_model_params_equal
from pyengine.utils.constants import N_VF_CAT, N_PLAYER
from pyengine import utils

pystratego = utils.get_pystratego()


@dataclass
class RLResumeConfig:
    kl_coef: Optional[float] = None
    temperature_coef: Optional[float] = None
    temperature_floor: Optional[float] = None
    lr_coef: Optional[float] = None
    lr_floor: Optional[float] = None
    adv_filt_thresh: Optional[float] = None
    arr_temperature_coef: Optional[float] = None
    arr_temperature_floor: Optional[float] = None
    eval_every: Optional[float] = None
    save_every: Optional[float] = None


@dataclass
class RLConfig:
    # shared params
    max_num_train: int = int(1e9)
    # rl params
    clip_range: float = 0.2
    ema_decay: float = 0.999
    td_lambda: float = 0.8
    gae_lambda: float = 0.5
    vf_coef: float = 1.0
    policy_coef: float = 1.0
    temperature_coef: float = 0.05
    temperature_ceil: float = 0.1
    temperature_floor: float = 0.001
    temperature_decay: float = 0.3
    kl_coef: float = 0.1
    adv_filt_rate: float = 0.75
    adv_filt_thresh: float = 0.01
    lr_coef: float = 0.5
    lr_decay: float = 1.1
    lr_ceil: float = 1e-4
    lr_floor: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 0.267
    # arrangement params
    arr_clip_range: float = 0.2
    arr_transformer: ArrangementTransformerConfig = field(
        default_factory=ArrangementTransformerConfig
    )
    arr_td_lambda: float = 1.0
    arr_gae_lambda: float = 1.0
    arr_vf_coef: float = 0.5
    arr_policy_coef: float = 1.0
    arr_reg_norm: float = 10.0
    arr_ent_pred_coef: float = 1.0
    arr_temperature_coef: float = 0.1
    arr_temperature_ceil: float = 1.0
    arr_temperature_floor: float = 0.001
    arr_temperature_decay: float = 0.3
    arr_kl_coef: float = 0.1
    arr_lr: float = 5e-5
    arr_weight_decay: float = 0.0
    arr_max_grad_norm: float = 0.5
    arr_batch_size: int = 1024
    arr_num_epoch_per_train: int = 5
    n_arr: int = 1024
    # others
    save_every: int = 2000
    eval_every: int = 2000
    dtype: str = "bfloat16"
    resume_from: str = ""  # weight file to resume from
    resume: RLResumeConfig = field(
        default_factory=RLResumeConfig
    )  # params to overwrite from resume
    # nets
    move_transformer: MoveTransformerConfig = field(default_factory=MoveTransformerConfig)
    # new params
    train_every_per_player: int = 101
    uniform_magnet: bool = True
    # compilation
    torch_compile: bool = True

    def get_dtype(self):
        if self.dtype == "float32":
            return torch.float32
        elif self.dtype == "bfloat16":
            return torch.bfloat16
        else:
            assert False, f"invalid type {self.dtype}"


class RL:
    def __init__(
        self,
        policy: MoveTransformer,
        env: Stratego,
        stats: utils.MultiCounter,
        save_dir: str,
        eval_manager: Optional[EvaluationManager],
        cfg: RLConfig,
        rank: int,
        arr_model: Optional[ArrangementTransformer] = None,
        saver_enabled: Optional[bool] = True,
    ):
        self.env = env
        self.cfg = cfg
        self.dtype = cfg.get_dtype()
        self.rank = rank
        self.save_dir = save_dir

        # track progress
        self.num_env_step = 0
        self.num_train_step = 0
        self.num_train_batch = 0
        self.num_train_transition = 0
        self.stats = stats
        self.eval_manager = eval_manager

        if cfg.torch_compile:
            policy = torch.compile(policy)

        self.container = TrainContainer(
            policy,
            power_schedule(
                cfg.lr_coef, self.num_train_step, cfg.lr_decay, cfg.lr_ceil, cfg.lr_floor
            ),
            cfg.weight_decay,
            cfg.resume_from,
        )
        self.ema_policy = EMA(self.container.net, cfg.ema_decay, cfg.resume_from)
        self.learner = self.container.net
        self.actor = self.container.net

        self.saver_enabled = saver_enabled
        self.saver = utils.Saver(save_dir, "model")
        if arr_model is not None:
            self.force_reset = True
            self.arr_container = TrainContainer(
                arr_model,
                cfg.arr_lr,
                cfg.arr_weight_decay,
                cfg.resume_from.replace("/model", "/arr_model"),
            )
            self.arrangement_actor = self.arr_container.net
            if cfg.torch_compile:
                self.arrangement_actor = torch.compile(self.arrangement_actor)
            self.arr_saver = utils.Saver(save_dir, "arr_model")
            self.arr_buffer = ArrangementBuffer(
                N_PLAYER * cfg.train_every_per_player + env.conf.max_num_moves,
                env.barrage,
                torch.device(self.rank),
                arr_model.cfg.use_cat_vf,
            )
            self.arr_ema_policy = EMA(
                self.arr_container.net,
                cfg.ema_decay,
                cfg.resume_from.replace("/model", "/arr_model"),
            )
        else:
            self.arr_container = None

        self.buffer = CircularBuffer(
            num_envs=env.num_envs,
            traj_len=env.conf.max_num_moves // N_PLAYER + cfg.train_every_per_player,
            train_every_per_player=cfg.train_every_per_player,
            use_cat_vf=policy.cfg.use_cat_vf,
            adv_filt_rate=cfg.adv_filt_rate,
            adv_filt_thresh=cfg.adv_filt_thresh,
            device=self.rank,
        )

        # flag used to declare that we've resumed
        self.resume_complete = False

        # set up learning
        self.learn_setup()

        # signal handling
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_sigusr1)

    def fast_forward_from_resume(self, weight_path):
        assert self.num_train_step == 0
        assert self.num_finished_games == 0

        resume_step = int(weight_path.split("/")[-1].split(".")[0][len("model") :])
        print(f"fast forward to num_train_step={resume_step}")

        log_file = os.path.join(os.path.dirname(weight_path), "log.pkl")
        # the pkl file consists of multiple pickled objects
        with open(log_file, "rb") as f:
            while True:
                try:
                    log_step = pickle.load(f)
                except EOFError:
                    break
                for k, v in log_step.items():
                    self.stats[k].append(v)

                num_train_step = int(log_step["count/num_train_step"])
                self.stats.summary(num_train_step, reset=True, print_msg=False)

                if num_train_step == resume_step:
                    self.num_train_step = resume_step
                    self.num_env_step = int(log_step["count/num_env_step"]) // self.env.num_envs
                    self.num_train_batch = int(log_step["count/num_train_batch"])
                    self.num_finished_games = int(log_step["count/num_finished_games"])
                    self.num_train_transition = int(log_step["count/num_train_transition"])
                    break

                if num_train_step > resume_step:
                    assert False, "cannot recover to the exact step, something wrong"

        print(f"fast forward done: {self.num_train_step=}, {self.num_finished_games=}")

    def learn(self):
        while self.num_train_step < self.cfg.max_num_train:
            is_terminating = self.should_terminate()

            with self.stopwatch.time("update_arrangements"):
                self.update_arrangements()

            if self.should_save() or is_terminating:
                self.save()

            if self.should_evaluate() or is_terminating:
                self.perform_evaluation()

            self.log_stats()

            if is_terminating:
                print("JOB COMPLETED SUCCESSFULLY")
                return

            self.collect_and_train()

    def learn_setup(self) -> None:
        self.stopwatch = utils.Stopwatch()
        self.num_train_step = 0
        self.num_finished_games = 0
        if self.cfg.resume_from:
            self.fast_forward_from_resume(self.cfg.resume_from)

    def collect_and_train(self) -> None:
        with self.stopwatch.time("rollout"), torch.no_grad(), utils.eval_mode(self.actor):
            self.collect_rollouts()

        with self.stopwatch.time("prep_data"):
            self.prepare_data()

        with self.stopwatch.time("train"):
            self.train()
        self.buffer.reset()

        if self.arr_container is not None:
            with self.stopwatch.time("arr_train"):
                self.arr_train()
            self.arr_buffer.filter(self.env.current_step)

        self.num_train_step += 1

    def should_evaluate(self) -> bool:
        return (
            (self.eval_manager is not None)
            and (self.num_train_step > 0)
            and (self.num_train_step % self.cfg.eval_every == 0)
        )

    def should_save(self) -> bool:
        return self.num_train_step % self.cfg.save_every == 0

    def save(self) -> None:
        if self.arr_container is not None:
            ema_tensor_arr, _, _, _, _, _ = generate_arrangements(
                self.cfg.n_arr, self.arr_ema_policy.ema_model
            )
        if not self.saver_enabled:
            return
        self.saver.save(
            self.container.net.state_dict(),
            self.ema_policy.ema_model.state_dict(),
            self.container.optim.state_dict(),
            self.num_train_step,
        )
        arrangements_file = f"{self.save_dir}/arrangements{self.num_train_step}.pkl"
        with open(arrangements_file, "wb") as f:
            pickle.dump(self.env.conf.initial_arrangements, f)
        ema_arrangements_file = f"{self.save_dir}/ema_arrangements{self.num_train_step}.pkl"
        if self.arr_container is not None:
            ema_tensor_arr, _, _, _, _, _ = generate_arrangements(
                self.cfg.n_arr, self.arr_ema_policy.ema_model
            )
            self.arr_saver.save(
                self.arr_container.net.state_dict(),
                self.arr_ema_policy.ema_model.state_dict(),
                self.arr_container.optim.state_dict(),
                self.num_train_step,
            )
            ema_string_arr = filter_terminal(
                to_string(
                    ema_tensor_arr,
                )
            )
            with open(ema_arrangements_file, "wb") as f:
                pickle.dump((ema_string_arr, ema_string_arr), f)
        else:
            with open(ema_arrangements_file, "wb") as f:
                pickle.dump(self.env.conf.initial_arrangements, f)

    def perform_evaluation(self) -> None:
        with self.stopwatch.time("eval"):
            eval_model = self.get_eval_model(ema=False)
            with torch.no_grad(), utils.eval_mode(eval_model):
                with autocast(device_type="cuda", dtype=self.dtype):
                    eval_data = self.eval_manager.evaluate(
                        eval_model,
                        self.env.conf.initial_arrangements,
                    )
            for key, val in eval_data.items():
                self.stats[f"eval/{key}"].append(val)

            if len(self.eval_manager.eval_against) > 0:  # skip arr generation if no eval_against
                if self.arr_container is None:
                    arr = self.env.conf.initial_arrangements
                else:
                    ema_tensor_arr, _, _, _, _, _ = generate_arrangements(
                        self.cfg.n_arr, self.arr_ema_policy.ema_model
                    )
                    ema_string_arr = filter_terminal(to_string(ema_tensor_arr))
                    arr = (ema_string_arr, ema_string_arr)
                ema_eval_model = self.get_eval_model(ema=True)
                with autocast(device_type="cuda", dtype=self.dtype):
                    with torch.no_grad(), utils.eval_mode(ema_eval_model):
                        eval_data_ema = self.eval_manager.evaluate(ema_eval_model, arr)
                for key, val in eval_data_ema.items():
                    self.stats[f"eval/ema_{key}"].append(val)

    def should_terminate(self) -> bool:
        if self.signal_received:
            print("Initiating terminal eval/save (reason: signal)")
            return True
        time_left = utils.get_slurm_remaining_time_minutes()
        if time_left is not None and time_left < 15:
            print("Initiating terminal eval/save (reason: time limit)")
            return True
        return False

    def log_stats(self) -> None:
        if self.num_train_step == 0 or not self.resume_complete:
            # we also want to skip if we have just resumed
            self.resume_complete = True
            return
        self.stats["count/num_train_step"].append(self.num_train_step)
        self.stats["count/num_train_batch"].append(self.num_train_batch)
        self.stats["count/num_train_transition"].append(self.num_train_transition)
        self.stats["count/num_env_step"].append(self.env.num_envs * self.num_env_step)
        self.num_finished_games += self.env.stats["num_finished_games"]
        self.stats["count/num_finished_games"].append(self.num_finished_games)
        self.stats["count/mem(GB)"].append(utils.get_mem_usage_gb())
        self.stats["count/gpu_mem(GB)"].append(utils.get_gpumem_usage_gb())
        for key, val in self.env.stats.items():
            self.stats[f"data/{key}"].append(val)
        time_data = self.stopwatch.summary()
        for k, v in time_data.items():
            self.stats[k].append(v)
        self.stats.summary(self.num_train_step, reset=True)
        self.env.reset_stats()

    def collect_rollouts(self):
        assert not self.actor.training

        while not self.buffer.ready_to_train():
            self.buffer.add_pre_act(
                step=self.env.current_step,
                num_moves=self.env.current_num_moves,
                legal_action_mask=self.env.current_legal_action_mask,
                is_terminal=self.env.current_is_terminal,
            )
            is_pl0_first_move = self.env.current_is_pl0_first_move
            is_pl1_first_move = self.env.current_is_pl1_first_move
            if self.env.current_step == 0:
                assert is_pl0_first_move.all()
            if self.env.current_step == 1:
                assert is_pl1_first_move.all()

            with autocast(device_type="cuda", dtype=self.dtype), utils.eval_mode(
                self.actor
            ), torch.no_grad():
                tensor_dict = self.actor(
                    infostate_tensor=self.env.current_infostate_tensor,
                    piece_ids=self.env.current_piece_ids,
                    legal_action_mask=self.env.current_legal_action_mask,
                    num_moves=self.env.current_num_moves,
                    acting_player=self.env.current_player,
                )
            actions = tensor_dict["action"]
            values = tensor_dict["value"]
            log_prob = tensor_dict["action_log_prob"]
            log_probs = tensor_dict["action_log_probs"]
            self.env.apply_actions(actions)
            self.stats["rollout/action_probs"].append(
                log_prob.exp()[~self.env.current_is_terminal].mean().item()
            )
            self.stats["rollout/entropy"].append(
                -log_prob[~self.env.current_is_terminal].mean().item()
            )

            # We store the reward for the player that acted just before us.
            rewards = self.env.current_reward_pl0
            if self.env.current_player == 0:
                rewards *= -1

            self.buffer.add_post_act(
                action=actions,
                value=values,
                log_prob=log_probs,
                reward=rewards,
                is_terminal=self.env.current_is_terminal,
            )
            if self.arr_container is not None:
                red_arr, blue_arr = self.env.current_zero_arrangements
                self.arr_buffer.add_rewards(
                    arrangements=red_arr,
                    is_newly_terminal=self.env.current_is_newly_terminal,
                    rewards=self.env.current_reward_pl0,
                )
                self.arr_buffer.add_rewards(
                    arrangements=blue_arr,
                    is_newly_terminal=self.env.current_is_newly_terminal,
                    rewards=-self.env.current_reward_pl0,
                )
            self.num_env_step += 1

    def prepare_data(self):
        buffer_info = self.buffer.process_data(self.cfg.td_lambda, self.cfg.gae_lambda)
        for k, v in buffer_info.items():
            self.stats[k].append(v)

        if self.arr_container is not None:
            arr_temperature = power_schedule(
                self.cfg.arr_temperature_coef,
                self.num_train_step,
                self.cfg.arr_temperature_decay,
                self.cfg.arr_temperature_ceil,
                self.cfg.arr_temperature_floor,
            )
            arr_info = self.arr_buffer.process_data(
                self.cfg.arr_td_lambda,
                self.cfg.arr_gae_lambda,
                arr_temperature,
                self.cfg.arr_reg_norm,
            )
            self.stats["arr_train/arr_temperature"].append(arr_temperature)
            if arr_info:
                for k, v in arr_info.items():
                    self.stats[k].append(v)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        assert self.container.net.training

        lr = power_schedule(
            self.cfg.lr_coef,
            self.num_train_step,
            self.cfg.lr_decay,
            self.cfg.lr_ceil,
            self.cfg.lr_floor,
        )
        for param_group in self.container.optim.param_groups:
            param_group["lr"] = lr

        temperature = power_schedule(
            self.cfg.temperature_coef,
            self.num_train_step,
            self.cfg.temperature_decay,
            self.cfg.temperature_ceil,
            self.cfg.temperature_floor,
        )

        clip_range = self.cfg.clip_range

        n_batch = self.buffer.n_batch()
        if dist.is_initialized():
            dist.all_reduce(n_batch, op=dist.ReduceOp.MIN)

        batch_sizes = []
        effective_batch_sizes = []
        for i, batch in enumerate(self.buffer.sample(self.env)):
            if i == n_batch:
                break

            assert batch.adv_mask.dtype == torch.bool
            assert batch.adv_mask.shape[0] == batch.infostates.shape[0]
            effective_batch_size = batch.adv_mask.sum().item()
            assert effective_batch_size > 0
            effective_batch_sizes.append(effective_batch_size)
            batch_size = math.prod(batch.infostates.shape[:-3])
            batch_sizes.append(batch_size)

            with autocast(device_type="cuda", dtype=self.dtype):
                assert batch.legal_actions.gather(-1, batch.actions.long().unsqueeze(-1)).all()
                tensor_dict = self.learner(
                    infostate_tensor=batch.infostates[batch.adv_mask],
                    piece_ids=batch.piece_ids[batch.adv_mask],
                    legal_action_mask=batch.legal_actions[batch.adv_mask],
                    num_moves=batch.num_moves[batch.adv_mask],
                )
                assert tensor_dict["value"].shape[0] == effective_batch_size
                assert tensor_dict["action_log_probs"].shape[0] == effective_batch_size
                batch = batch.apply_mask()
                log_probs = tensor_dict["action_log_probs"]
                values_pred = tensor_dict["value"]
                advantages = batch.advantages

                old_log_prob = batch.log_probs.gather(
                    -1, batch.actions.long().unsqueeze(-1)
                ).squeeze(-1)
                log_prob = log_probs.gather(-1, batch.actions.long().unsqueeze(-1)).squeeze(-1)
                ratio = torch.exp(log_prob - old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                probs = log_probs.exp() * batch.legal_actions
                kl_loss = (probs * (log_probs - batch.log_probs)).sum(dim=-1).mean()

                assert batch.values.shape == values_pred.shape == batch.returns.shape
                if self.container.net.cfg.use_cat_vf:
                    log_cur = torch.log_softmax(values_pred, dim=-1)
                    log_prev = torch.log_softmax(batch.values, dim=-1)
                    value_loss = -(batch.returns * log_cur).sum(-1).mean()
                    vf_change = (log_cur.exp() * (log_cur - log_prev)).sum(-1).mean()
                else:
                    value_loss = F.mse_loss(values_pred, batch.returns, reduction="none").mean()
                    vf_change = F.mse_loss(values_pred, batch.values, reduction="none").mean()

                entropy = -(probs * log_probs).sum(-1)
                if self.cfg.uniform_magnet:
                    magnet = batch.legal_actions.float() / batch.legal_actions.sum(-1, keepdim=True)
                else:
                    magnet = utils.helper.get_weighted_uniform_policy(batch.legal_actions)
                xe = -(probs * torch.log(magnet.clamp(min=1e-10))).sum(dim=-1)
                magnet_kl = -(entropy - xe).mean()

                loss = (
                    self.cfg.policy_coef * policy_loss
                    + temperature * magnet_kl
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.kl_coef * kl_loss
                )
            self.container.optim.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(  # type: ignore
                self.container.net.parameters(), self.cfg.max_grad_norm
            )
            self.container.optim.step()

            self.stats["train/action_probs"].append(old_log_prob.exp().mean().item())
            self.stats["train/advantages"].append(advantages.mean().item())
            self.stats["train/batch_size"].append(batch_size)
            self.stats["train/effective_batch_size"].append(effective_batch_size)
            self.stats["train/value_loss"].append(value_loss.item())
            self.stats["train/policy_loss"].append(policy_loss.item())
            self.stats["train/vf_change"].append(vf_change.item())
            self.stats["train/magnet_kl"].append(magnet_kl.item())
            self.stats["train/kl_loss"].append(kl_loss.item())
            self.stats["train/clip_fraction"].append(
                ((ratio - 1).abs() > clip_range).float().mean().item()
            )
            self.num_train_batch += 1
            self.stats["train/g_norm"].append(grad_norm.item())
            self.stats["train/lr"].append(self.container.optim.param_groups[0]["lr"])
            self.stats["train/loss"].append(loss.item())

        if len(batch_sizes) > 0:
            self.stats["train/temperature"].append(temperature)
            self.stats["train/min_batch_size"].append(min(batch_sizes))
            self.stats["train/max_batch_size"].append(max(batch_sizes))
            self.stats["train/num_batches"].append(len(batch_sizes))
            self.stats["train/min_effective_batch_size"].append(min(effective_batch_sizes))
            self.stats["train/max_effective_batch_size"].append(max(effective_batch_sizes))
            self.num_train_transition += sum(batch_sizes)
            self.ema_policy.update()
            if dist.is_initialized():
                check_model_params_equal(self.container.net)

    def arr_train(self) -> None:
        assert self.arr_container is not None
        assert self.arr_container.net.training

        clip_range = self.cfg.arr_clip_range

        rank_examples = self.arr_buffer.ready_flags.sum()
        if dist.is_initialized():
            dist.all_reduce(rank_examples, op=dist.ReduceOp.MIN)
        num_batches = (rank_examples + self.cfg.arr_batch_size - 1) // self.cfg.arr_batch_size

        for _ in range(self.cfg.arr_num_epoch_per_train):
            for batch_num, batch in enumerate(self.arr_buffer.sample(self.cfg.arr_batch_size)):
                # Stop epoch early if one rank has fewer examples
                if batch_num == num_batches:
                    break
                with autocast(device_type="cuda", dtype=self.dtype):
                    tensor_dict = self.arr_container.net(batch.arrangements)
                    logits, values_pred, regs_pred = (
                        tensor_dict["logits"],
                        tensor_dict["value"],
                        tensor_dict["ent_pred"],
                    )
                    log_probs = F.log_softmax(logits, dim=-1)

                    advantages = batch.advantages

                    log_prob = log_probs.gather(-1, batch.arrangements.argmax(-1, True)).squeeze(-1)
                    old_log_prob = batch.log_probs.gather(
                        -1, batch.arrangements.argmax(-1, True)
                    ).squeeze(-1)
                    ratio = torch.exp(log_prob - old_log_prob)

                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    kl_loss = (log_probs.exp() * (log_probs - batch.log_probs)).sum(dim=-1).mean()

                    if not self.arr_container.net.cfg.use_cat_vf:
                        values_pred = values_pred.squeeze(-1)
                        assert values_pred.dim() == 2
                        assert batch.returns.size() == values_pred.size()

                        value_loss = F.mse_loss(values_pred, batch.returns, reduction="none").mean()
                    else:
                        assert values_pred.dim() == 3 and values_pred.shape[-1] == N_VF_CAT
                        assert batch.returns.size() == values_pred.size()
                        value_loss = (
                            -(batch.returns * torch.log_softmax(values_pred, dim=-1)).sum(-1).mean()
                        )
                        arr_idx = batch.arrangements.argmax(-1)
                        B, K = arr_idx.shape

                    assert regs_pred.dim() == 3 and regs_pred.shape[-1] == 1
                    regs_pred = regs_pred.squeeze(-1)
                    entropy_loss = F.mse_loss(regs_pred, batch.reg_returns, reduction="none").mean()

                    loss = (
                        self.cfg.arr_policy_coef * policy_loss
                        + self.cfg.arr_ent_pred_coef * entropy_loss
                        + self.cfg.arr_vf_coef * value_loss
                        + self.cfg.arr_kl_coef * kl_loss
                    )

                self.arr_container.optim.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.arr_container.net.parameters(),
                    self.cfg.arr_max_grad_norm,
                )
                self.arr_container.optim.step()

                self.num_train_batch += 1
                self.stats["arr_train/policy_loss"].append(policy_loss.item())
                self.stats["arr_train/value_loss"].append(value_loss.item())
                self.stats["arr_train/entropy_loss"].append(entropy_loss.item())
                self.stats["arr_train/entropy"].append(-log_prob.sum(dim=-1).mean().item())
                self.stats["arr_train/g_norm"].append(grad_norm)
                self.stats["arr_train/kl_loss"].append(kl_loss.item())
                self.stats["arr_train/clip_fraction"].append(
                    ((ratio - 1).abs() > clip_range).float().mean().item()
                )
                self.stats["arr_train/lr"].append(self.arr_container.optim.param_groups[0]["lr"])
        self.arr_ema_policy.update()
        if dist.is_initialized():
            check_model_params_equal(self.arr_container.net)

    def update_arrangements(self) -> None:
        if not self.arr_container:
            return

        assert self.arr_container is not None
        assert self.arr_container.net is not None

        # Generate arrangements and add them to arrangement buffer
        (
            tensor_arrs,
            values,
            reg_values,
            log_probs,
            needs_flip,
            piece_info,
        ) = generate_arrangements(self.cfg.n_arr, self.arrangement_actor)
        for k, v in piece_info.items():
            self.stats[k].append(v)
        self.arr_buffer.add_arrangements(
            tensor_arrs,
            values,
            reg_values,
            log_probs,
            needs_flip,
            self.env.current_step,
        )
        string_arrs = filter_terminal(to_string(tensor_arrs))
        # Update the initial arrangement distribution
        self.env.change_reset_behavior_to_random_initial_arrangement(
            [string_arrs[::2], string_arrs[1::2]]  # Even arrangements for red, odd for blue
        )

        # If we haven't started training yet, do a reset to get the initial distribution right
        if self.force_reset:
            self.force_reset = False
            self.env.reset()

    def handle_sigusr1(self, signum, frame):
        try:
            print("Received SIGUSR1")
            self.signal_received = True
        except Exception as e:
            print(f"Error in signal handler: {e}")

    def get_eval_model(self, ema: bool = False) -> torch.nn.Module:
        if ema:
            model = self.ema_policy.ema_model
        else:
            model = self.container.net
        return model


def power_schedule(coef: float, step: int, decay: float, ceil: float, floor: float) -> float:
    x = coef / ((step + 1) ** decay)
    x = max(x, floor)
    x = min(x, ceil)
    return x
