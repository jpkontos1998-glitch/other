"""A python wrapper for pystratego with handy utilities"""

from collections import defaultdict
from typing import Any, Optional, Sequence

import torch

from pyengine import utils
from pyengine.utils.types import GenerateArgType
from pyengine.utils.constants import NOTMOVE_ID, N_BOARD_CELL

pystratego = utils.get_pystratego()


class Stratego:
    """
    Light wrapper around the `StrategoRolloutBuffer` backend.


    Tensor output lifetime
    ======================

    The `StrategoRolloutBuffer` backend makes the following guarantees about the
    returned tensors:
      - Expensive tensors (those that are computed via `compute_*` methods) are
        valid until the next `compute_*` method is called.
      - Inexpensive tensor (those that are returned by `get_*` methods) are valid
        until the next actions are supplied (by the `apply_actions` method).
        In practice, this is usually a conservative bound, though the issue (which
        would potentially be tricky to debug) is that the StrategoRolloutBuffer
        backend uses a circular buffer, and so tensor memory can be rewritten after
        enough actions are taken.

    So, overall, the tensors returned by the backend have a rather short lifetime.
    The recommended approach to not incur into problem is to query the tensors on
    demand rather then storing the pointers.

    In order to avoid issues, the python wrapper on the backend does not leak
    tensors returned by the backend, and rather clones them.
    """

    def __init__(
        self,
        num_envs: int,
        traj_len_per_player: int,
        full_info: bool = False,
        barrage: bool = False,
        custom_arrs: Optional[Sequence[Sequence[str]]] = None,
        move_memory: int = 86,
        max_num_moves: int = 4_000,
        max_num_moves_between_attacks: int = 200,
        **kwargs,
    ):
        self.num_envs = num_envs
        # the plus 2 accouts for storing the additional obs
        # after applying traj_len_per_player * 2 actions
        # and the num_row must be even number
        if kwargs.get("nonsteppable", False):
            self.num_row = traj_len_per_player * 2 + 2
        else:
            self.num_row = max(210, traj_len_per_player * 2 + 2)
        self.traj_len_per_player = traj_len_per_player
        self.move_memory = move_memory

        # Note that it is possible to change whether an env is barrage or full info
        # using the change reset functionalities. While, ideally, this will never
        # happen, we are not currently enforcing that it doesn't. Thus, it is only
        # guaranteed that the attributes below accurately reflect the environment
        # properties at the time of initialization.
        self.barrage = barrage
        self.full_info = full_info
        if barrage:
            self.n_piece_per_player = 8
        else:
            self.n_piece_per_player = 40

        if "reset_behavior" not in kwargs:
            if custom_arrs is None:
                if full_info and barrage:
                    reset_behavior = pystratego.ResetBehavior.FULLINFO_RANDOM_JB_BARRAGE_BOARD
                elif full_info and not barrage:
                    reset_behavior = pystratego.ResetBehavior.FULLINFO_RANDOM_JB_CLASSIC_BOARD
                elif not full_info and barrage:
                    reset_behavior = pystratego.ResetBehavior.RANDOM_JB_BARRAGE_BOARD
                else:
                    reset_behavior = pystratego.ResetBehavior.RANDOM_JB_CLASSIC_BOARD
            else:
                if full_info:
                    reset_behavior = (
                        pystratego.ResetBehavior.FULLINFO_RANDOM_CUSTOM_INITIAL_ARRANGEMENT
                    )
                else:
                    reset_behavior = pystratego.ResetBehavior.RANDOM_CUSTOM_INITIAL_ARRANGEMENT
        else:
            assert custom_arrs is None, "Cannot specify both custom_arrs and reset_behavior"
            reset_behavior = kwargs["reset_behavior"]
            kwargs.pop("reset_behavior")

        self.env = pystratego.StrategoRolloutBuffer(
            self.num_row + move_memory,
            self.num_envs,
            reset_behavior=reset_behavior,
            initial_arrangements=custom_arrs,
            move_memory=move_memory,
            max_num_moves=max_num_moves,
            max_num_moves_between_attacks=max_num_moves_between_attacks,
            **kwargs,
        )
        if "verbose" in kwargs and kwargs["verbose"]:
            print("*** VERBOSE MODE ACTIVE ***")
            self.env = utils.VerboseShim(self.env)

        self.obs_shape = self.env.infostate_tensor.size()[1:]
        self.action_dim = self.env.legal_action_mask.size(1)
        self.device = self.env.infostate_tensor.device
        self.conf = self.env.conf

        # prepare the observation, since we do not have an explicit reset function
        self.stats = defaultdict(float)

        self._invalidate_cache()

    def _invalidate_cache(self):
        # FIXME(gfarina): Move the caching logic to the backend?
        self.computed_infostate_tensor_step = None
        self.computed_legal_action_mask_step = None
        self.computed_reward_pl0_step = None
        self.computed_is_unknown_piece_step = None
        self.computed_piece_type_onehot_step = None
        self.computed_unknown_piece_type_onehot_step = None
        self.computed_unknown_piece_has_moved_step = None
        self.computed_unknown_piece_position_onehot_step = None

    def seed_action_sampler(self, seed) -> None:
        self.env.seed_action_sampler(seed)

    def apply_actions(self, actions: torch.Tensor) -> None:
        self.env.apply_actions(actions)

        (
            is_newly_terminal,
            is_gamelen_timeout,
            is_battle_timeout,
            is_kamikaze,
            is_flag_capture,
            is_wipe_out,
        ) = self.current_termination_reason
        num_finished_games = is_newly_terminal.sum().item()
        num_battle_timeout = is_battle_timeout.sum().item()
        num_gamelen_timeout = is_gamelen_timeout.sum().item()
        num_kamikaze_games = is_kamikaze.sum().item()
        num_flag_capture_games = is_flag_capture.sum().item()
        num_wipe_out_games = is_wipe_out.sum().item()
        assert (
            num_finished_games
            == num_battle_timeout
            + num_gamelen_timeout
            + num_kamikaze_games
            + num_flag_capture_games
            + num_wipe_out_games
        )

        reward_pl0 = self.current_reward_pl0
        reward_sum = (is_newly_terminal * reward_pl0).sum().item()

        if num_finished_games == 0:
            return
        self.stats["pl0_return"] = (
            self.stats["pl0_return"] * self.stats["num_finished_games"] + reward_sum
        ) / (self.stats["num_finished_games"] + num_finished_games)
        self.stats["gamelen"] = (
            self.stats["gamelen"] * self.stats["num_finished_games"]
            + self.current_num_moves[is_newly_terminal].sum().item()
        ) / (self.stats["num_finished_games"] + num_finished_games)
        self.stats["tie_prob"] = (
            self.stats["tie_prob"] * self.stats["num_finished_games"]
            + (num_battle_timeout + num_gamelen_timeout + num_kamikaze_games)
        ) / (self.stats["num_finished_games"] + num_finished_games)
        self.stats["gamelen_timeout_prob"] = (
            self.stats["gamelen_timeout_prob"] * self.stats["num_finished_games"]
            + (num_gamelen_timeout)
        ) / (self.stats["num_finished_games"] + num_finished_games)
        self.stats["num_finished_games"] += num_finished_games
        self.stats["num_gamelen_timeout"] += num_gamelen_timeout
        self.stats["num_battle_timeout"] += num_battle_timeout
        self.stats["num_kamikaze"] += num_kamikaze_games
        self.stats["num_flag_capture"] += num_flag_capture_games
        self.stats["num_wipe_out"] += num_wipe_out_games

    def infostate_tensor(self, step: int) -> torch.Tensor:
        if self.computed_infostate_tensor_step != step:
            self.env.compute_infostate_tensor(step)
            self.computed_infostate_tensor_step = step
        return self.env.infostate_tensor.clone()

    def legal_action_mask(self, step: int) -> torch.Tensor:
        if self.computed_legal_action_mask_step != step:
            self.env.compute_legal_action_mask(step)
            self.computed_legal_action_mask_step = step
        return self.env.legal_action_mask.clone()

    def reward_pl0(self, step: int) -> torch.Tensor:
        if self.computed_reward_pl0_step != step:
            self.env.compute_reward_pl0(step)
            self.computed_reward_pl0_step = step
        return self.env.reward_pl0.clone()

    def sample_random_legal_action(self, out=None):
        if out is None:
            out = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.env.sample_random_legal_action(out)
        return out

    def sample_first_legal_action(self, out=None):
        if out is None:
            out = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.env.sample_first_legal_action(out)
        return out

    def is_unknown_piece(self, step: int) -> torch.Tensor:
        if self.computed_is_unknown_piece_step != step:
            self.env.compute_is_unknown_piece(step)
            self.computed_is_unknown_piece_step = step
        return self.env.is_unknown_piece.clone()

    def piece_type_onehot(self, step: int) -> torch.Tensor:
        if self.computed_piece_type_onehot_step != step:
            self.env.compute_piece_type_onehot(step)
            self.computed_piece_type_onehot_step = step
        return self.env.piece_type_onehot.clone()

    def unknown_piece_type_onehot(self, step: int) -> torch.Tensor:
        if self.computed_unknown_piece_type_onehot_step != step:
            self.env.compute_unknown_piece_type_onehot(step, self.n_piece_per_player)
            self.computed_unknown_piece_type_onehot_step = step
        return self.env.unknown_piece_type_onehot.clone()

    def unknown_piece_has_moved(self, step: int) -> torch.Tensor:
        if self.computed_unknown_piece_has_moved_step != step:
            self.env.compute_unknown_piece_has_moved(step, self.n_piece_per_player)
            self.computed_unknown_piece_has_moved_step = step
        return self.env.unknown_piece_has_moved.clone()

    def unknown_piece_position_onehot(self, step: int) -> torch.Tensor:
        if self.computed_unknown_piece_position_onehot_step != step:
            self.env.compute_unknown_piece_position_onehot(step, self.n_piece_per_player)
            self.computed_unknown_piece_position_onehot_step = step
        return self.env.unknown_piece_position_onehot.clone()

    def unknown_piece_counts(self, step: int) -> torch.Tensor:
        # Player 0's hidden piece counts are at indices 1600:1612
        # Player 1's hidden piece counts are at indices 1612:1624
        if self.acting_player(step) == 0:
            return self.env.get_board_tensor(step)[:, 1612:1624].clone()
        else:
            return self.env.get_board_tensor(step)[:, 1600:1612].clone()

    def is_terminal(self, step: int) -> torch.Tensor:
        # > allocates new memory
        return self.env.get_terminated_since(step) > 0

    def is_newly_terminal(self, step: int) -> torch.Tensor:
        # == allocates new memory
        return self.env.get_terminated_since(step) == 1

    def num_moves(self, step: int) -> torch.Tensor:
        return self.env.get_num_moves(step).clone()

    def num_moves_since_reset(self, step: int) -> torch.Tensor:
        return self.env.get_num_moves_since_reset(step).clone()

    def termination_reason(
        self, step: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        is_newly_terminal = self.is_newly_terminal(step)
        num_moves = self.num_moves(step)
        num_moves_last_attack = self.num_moves_since_last_attack(step)
        flag_captured = self.env.get_flag_captured(step)
        reward_pl0 = self.reward_pl0(step)
        is_zero_reward = reward_pl0 == 0
        is_non_zero_reward = ~is_zero_reward
        is_gamelen_timeout = is_newly_terminal & (
            num_moves == self.conf.max_num_moves + 1
        )  # time out triggers on step max_num_moves+1
        is_battle_timeout = (
            is_newly_terminal
            & (num_moves_last_attack == self.conf.max_num_moves_between_attacks + 1)
            & (~is_gamelen_timeout)
        )  # battle timeout triggers on step max_num_moves_between_attacks+1
        is_kamikaze = (
            is_newly_terminal
            & (num_moves <= self.conf.max_num_moves)
            & (num_moves_last_attack <= self.conf.max_num_moves_between_attacks)
            & is_zero_reward
        )
        is_flag_capture = is_newly_terminal & flag_captured & is_non_zero_reward
        is_wipe_out = is_newly_terminal & ~flag_captured & is_non_zero_reward
        return (
            is_newly_terminal,
            is_gamelen_timeout,
            is_battle_timeout,
            is_kamikaze,
            is_flag_capture,
            is_wipe_out,
        )

    def played_actions(self, step: int) -> torch.Tensor:
        return self.env.get_played_actions(step).clone()

    def move_summary(self, step: int) -> torch.Tensor:
        return self.env.get_move_summary(step).clone()

    def last_two_moves(self, step: int) -> torch.Tensor:
        """
        Returns the last two moves made in the environment at the given step.
        The output is a tensor of shape (num_envs, 12), representing concatenation
        of two move_summary (previous, previous previous).

        When data is not available (first move of an env), zeros are used as fillers.
        """

        previous, previous_previous = None, None
        if step > 0:
            previous = self.env.get_move_summary(step - 1).clone()
            assert (previous[:, 0:2] <= 99).all()
            previous[:, 0:2] = 99 - previous[:, 0:2]  # Relativize to player at `step`
            is_terminated = self.is_terminal(step - 1)
            previous[is_terminated] = NOTMOVE_ID
            piece_ids = previous[:, 4]
            needs_inversion = piece_ids < N_BOARD_CELL
            piece_ids[needs_inversion] = 99 - piece_ids[needs_inversion]
            previous[:, 4] = piece_ids
        else:
            previous = torch.full(
                (self.num_envs, 6),
                NOTMOVE_ID,
                dtype=torch.uint8,
                device=f"cuda:{self.env.conf.cuda_device}",
            )

        if step > 1:
            previous_previous = self.env.get_move_summary(step - 2).clone()
            is_terminated = self.is_terminal(step - 2)
            previous_previous[is_terminated] = NOTMOVE_ID
            piece_ids = previous_previous[:, 5]
            needs_inversion = piece_ids < N_BOARD_CELL
            piece_ids[needs_inversion] = 99 - piece_ids[needs_inversion]
            previous_previous[:, 5] = piece_ids
        else:
            previous_previous = torch.full(
                (self.num_envs, 6),
                NOTMOVE_ID,
                dtype=torch.uint8,
                device=f"cuda:{self.env.conf.cuda_device}",
            )

        # Here, we account for the prehistory.
        if self.env.conf.reset_behavior == pystratego.ResetBehavior.CUSTOM_ENV_STATE:
            assert self.env.conf.reset_state is not None
            assert self.env.conf.reset_state.move_summary_history.size(0) >= 2
            nmsr = self.num_moves_since_reset(step)
            nm = self.num_moves(step)
            prehistory_last = self.env.conf.reset_state.move_summary_history[-1]
            prehistory_second_last = self.env.conf.reset_state.move_summary_history[-2]

            # Every time nmsr = 0 but nm >= 1, we load the last action from the prehistory
            # into the first position.
            mask = (nmsr == 0) & (nm >= 1)
            previous[mask] = prehistory_last[mask]
            previous[mask, 0:2] = 99 - previous[mask, 0:2]  # Relativize to player at `step`
            piece_ids = previous[mask, 4]
            needs_inversion = piece_ids < N_BOARD_CELL
            piece_ids[needs_inversion] = 99 - piece_ids[needs_inversion]
            previous[mask, 4] = piece_ids

            # Every time nmsr = 1 but nm >= 2, we load the last action from the
            # prehistory into the second position.
            mask = (nmsr == 1) & (nm >= 2)
            previous_previous[mask] = prehistory_last[mask]
            piece_ids = previous_previous[mask, 5]
            needs_inversion = piece_ids < N_BOARD_CELL
            piece_ids[needs_inversion] = 99 - piece_ids[needs_inversion]
            previous_previous[mask, 5] = piece_ids

            # Every time nmsr = 0 but nm >= 2, we load the second last action from the
            # prehistory into the second position.
            mask = (nmsr == 0) & (nm >= 2)
            previous_previous[mask] = prehistory_second_last[mask]
            piece_ids = previous_previous[mask, 5]
            needs_inversion = piece_ids < N_BOARD_CELL
            piece_ids[needs_inversion] = 99 - piece_ids[needs_inversion]
            previous_previous[mask, 5] = piece_ids

        return torch.cat((previous, previous_previous), dim=1).long()

    def acting_player(self, step: int) -> int:
        return self.env.acting_player(step)

    def is_first_pl0_move(self, step: int) -> torch.Tensor:
        return self.num_moves_since_reset(step) == 0

    def is_first_pl1_move(self, step: int) -> torch.Tensor:
        return self.num_moves_since_reset(step) == 1

    def snapshot_state(self, step: int) -> pystratego.EnvState:
        # There is no need to `clone` here, as `snapshot_state` returns
        # an owned view over the env state.
        return self.env.snapshot_state(step)

    def snapshot_env_history(self, step: int, env: int) -> list[pystratego.EnvState]:
        return self.env.snapshot_env_history(step, env)

    def board_strs_pretty(self, step: int) -> list[str]:
        return pretty_boards(self.board_strs(step))

    def board_strs(self, step: int) -> list[str]:
        return self.env.board_strs(step)

    def zero_board_strs(self, step: int) -> list[str]:
        return self.env.zero_board_strs(step)

    def zero_boards(self, step: int) -> torch.Tensor:
        return self.env.get_zero_board_tensor(step)

    def zero_arrangements(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """NOTE: Assumes that zero_board is an initial board"""
        boards = self.snapshot_state(step).zero_boards[:, :1600:16]
        # First four bits of each byte are the piece values
        piece_vals = boards & 15
        # Red places first 40 pieces
        red = piece_vals[:, :40]
        # Blue places last 40 pieces; flip to match pystratego convention
        blue = piece_vals[:, 60:100].flip(dims=[1])
        return red, blue

    def change_reset_behavior_to_env_state(self, env_state: Optional[pystratego.EnvState]):
        self._invalidate_cache()
        self.env.change_reset_behavior(env_state)

    def change_reset_behavior_to_initial_board(self, init_board: str):
        assert len(init_board) == 100
        init_board = init_board.upper()
        self._invalidate_cache()
        arrs = utils.init_helpers.initial_boards_to_arrangements([init_board])
        self.change_reset_behavior_to_random_initial_arrangement(arrs)

    def change_reset_behavior_to_replay(self, init_boards: Sequence[str]):
        assert all(len(board) == 100 for board in init_boards)
        init_boards = [board.upper() for board in init_boards]
        self._invalidate_cache()
        arrs = utils.init_helpers.initial_boards_to_arrangements(init_boards)
        self.env.change_reset_behavior(arrs, randomize=False)

    def change_reset_behavior_to_random_initial_arrangement(
        self, initial_arrangements: Sequence[list[str]]
    ):
        assert len(initial_arrangements) == 2
        self._invalidate_cache()
        self.env.change_reset_behavior(initial_arrangements)

    def change_reset_behavior(
        self,
        reset_behavior: pystratego.ResetBehavior,
        randomize: bool = True,
        fullinfo: bool = False,
    ):
        self._invalidate_cache()
        self.env.change_reset_behavior(reset_behavior, randomize=randomize, fullinfo=fullinfo)

    def piece_ids(self, step: int) -> torch.Tensor:
        # board_tensor is [envs, 1920]
        board_tensor = self.env.get_board_tensor(step)
        assert board_tensor.shape == (self.num_envs, 1920)
        # every other element is the piece id starting from 1
        colors = board_tensor[:, 0:1600:16].reshape(-1, 10, 10).bitwise_and(0b00110000) / 16
        piece_ids = board_tensor[:, 1:1600:16].reshape(-1, 10, 10).clone()
        if self.env.acting_player(step) == 0:
            blue_pieces = colors == 2
            piece_ids[blue_pieces] *= -1
            piece_ids[blue_pieces] += 99
            piece_ids = piece_ids.clone()
        else:
            red_pieces = colors == 1
            piece_ids[red_pieces] *= -1
            piece_ids[red_pieces] += 99
            piece_ids = piece_ids.flip(-1, -2)
        return piece_ids

    def save_games(self, path):
        self.env.save_games(path)

    def stop_saving_games(self):
        self.env.stop_saving_games()

    def num_moves_since_last_attack(self, step: int) -> torch.Tensor:
        return self.env.get_num_moves_since_last_attack(step).clone()

    def generate_args(self, step: int, env: int, generate_arg_type: GenerateArgType) -> Any:
        unknown_piece_position_onehot = self.unknown_piece_position_onehot(step)[env]
        unknown_piece_has_moved = self.unknown_piece_has_moved(step)[env]
        unknown_piece_counts = self.unknown_piece_counts(step)[env]
        if generate_arg_type == GenerateArgType.MARGLIZED_UNIFORM:
            infostate_tensor = self.infostate_tensor(step)[env]
            return (
                unknown_piece_position_onehot,
                unknown_piece_has_moved,
                unknown_piece_counts,
                infostate_tensor,
            )
        elif generate_arg_type == GenerateArgType.UNIFORM:
            return unknown_piece_has_moved, unknown_piece_counts
        elif generate_arg_type == GenerateArgType.PLANAR_TRANSFORMER:
            infostate_tensor = self.infostate_tensor(step)[env]
            piece_ids = self.piece_ids(step)[env]
            return (
                unknown_piece_position_onehot,
                unknown_piece_has_moved,
                unknown_piece_counts,
                infostate_tensor,
                piece_ids,
            )
        assert generate_arg_type == GenerateArgType.TEMPORAL_TRANSFORMER
        even_states, odd_states = self.snapshot_env_history(step, env)
        if self.acting_player(step) == 0:
            states = even_states
        else:
            states = odd_states
        auxiliary_env = Stratego(
            states.num_envs,
            2,
            quiet=2,
            reset_state=states,
            reset_behavior=pystratego.ResetBehavior.CUSTOM_ENV_STATE,
            max_num_moves_between_attacks=self.conf.max_num_moves_between_attacks,
            max_num_moves=self.conf.max_num_moves,
            nonsteppable=True,
            cuda_device=self.conf.cuda_device,
        )
        return (
            unknown_piece_position_onehot,
            unknown_piece_has_moved,
            unknown_piece_counts,
            auxiliary_env.current_infostate_tensor,
            auxiliary_env.current_piece_ids,
            auxiliary_env.current_num_moves,
        )

    @property
    def INFOSTATE_CHANNEL_DESCRIPTION(self) -> list[str]:
        return self.env.INFOSTATE_CHANNEL_DESCRIPTION

    @property
    def current_step(self) -> int:
        return self.env.current_step()

    @property
    def current_player(self) -> int:
        return self.env.current_player()

    @property
    def current_infostate_tensor(self) -> torch.Tensor:
        return self.infostate_tensor(self.current_step)

    @property
    def current_legal_action_mask(self) -> torch.Tensor:
        return self.legal_action_mask(self.current_step)

    @property
    def current_reward_pl0(self) -> torch.Tensor:
        return self.reward_pl0(self.current_step)

    @property
    def current_is_unknown_piece(self) -> torch.Tensor:
        return self.is_unknown_piece(self.current_step)

    @property
    def current_piece_type_onehot(self) -> torch.Tensor:
        return self.piece_type_onehot(self.current_step)

    @property
    def current_unknown_piece_type_onehot(self) -> torch.Tensor:
        return self.unknown_piece_type_onehot(self.current_step)

    @property
    def current_unknown_piece_has_moved(self) -> torch.Tensor:
        return self.unknown_piece_has_moved(self.current_step)

    @property
    def current_unknown_piece_position_onehot(self) -> torch.Tensor:
        return self.unknown_piece_position_onehot(self.current_step)

    @property
    def current_unknown_piece_counts(self) -> torch.Tensor:
        return self.unknown_piece_counts(self.current_step)

    @property
    def current_is_terminal(self) -> torch.Tensor:
        return self.is_terminal(self.current_step)

    @property
    def current_is_newly_terminal(self) -> torch.Tensor:
        return self.is_newly_terminal(self.current_step)

    @property
    def current_is_pl0_first_move(self) -> torch.Tensor:
        return self.is_first_pl0_move(self.current_step)

    @property
    def current_is_pl1_first_move(self) -> torch.Tensor:
        return self.is_first_pl1_move(self.current_step)

    @property
    def current_has_just_reset(self) -> torch.Tensor:
        return self.current_num_moves_since_reset == 0

    @property
    def current_played_actions(self) -> torch.Tensor:
        return self.played_actions(self.current_step)

    @property
    def current_acting_player(self) -> int:
        return self.acting_player(self.current_step)

    @property
    def current_num_moves(self) -> torch.Tensor:
        return self.num_moves(self.current_step)

    @property
    def current_termination_reason(self) -> tuple:
        return self.termination_reason(self.current_step)

    @property
    def current_num_moves_since_reset(self) -> torch.Tensor:
        return self.num_moves_since_reset(self.current_step)

    @property
    def current_state(self) -> pystratego.EnvState:
        return self.snapshot_state(self.current_step)

    @property
    def current_board_strs(self) -> list[str]:
        return self.board_strs(self.current_step)

    @property
    def current_board_strs_pretty(self) -> list[str]:
        return pretty_boards(self.current_board_strs)

    @property
    def current_zero_board_strs(self) -> list[str]:
        return self.zero_board_strs(self.current_step)

    @property
    def current_move_summary(self) -> torch.Tensor:
        return self.move_summary(self.current_step)

    @property
    def current_last_two_moves(self) -> torch.Tensor:
        return self.last_two_moves(self.current_step)

    @property
    def current_zero_boards(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.zero_boards(self.current_step)

    @property
    def current_zero_arrangements(self) -> tuple[torch.Tensor, torch.Tensor]:
        """NOTE: Assumes that zero_board is an initial board"""
        return self.zero_arrangements(self.current_step)

    @property
    def current_piece_ids(self) -> torch.Tensor:
        return self.piece_ids(self.current_step)

    @property
    def current_num_moves_since_last_attack(self) -> torch.Tensor:
        return self.num_moves_since_last_attack(self.current_step)

    def reset(self) -> None:
        self._invalidate_cache()
        self.env.reset()

    def reset_stats(self) -> None:
        self.stats.clear()


def pretty_boards(boards: list[str]) -> list[str]:
    assert isinstance(boards, list)
    return list(map(pretty_board, boards))


def pretty_board(board: str) -> str:
    assert (
        isinstance(board, str) and len(board) == 200 and all([c in (".", "@") for c in board[1::2]])
    )
    s = board.replace("@", "").replace(".", "")
    s = s.replace("a", ".")
    return "\n".join(
        reversed(
            [
                "abs_row 0 | " + s[:10],
                "abs_row 1 | " + s[10:20],
                "abs_row 2 | " + s[20:30],
                "abs_row 3 | " + s[30:40],
                "abs_row 4 | " + s[40:50],
                "abs_row 5 | " + s[50:60],
                "abs_row 6 | " + s[60:70],
                "abs_row 7 | " + s[70:80],
                "abs_row 8 | " + s[80:90],
                "abs_row 9 | " + s[90:100],
            ]
        )
    )
