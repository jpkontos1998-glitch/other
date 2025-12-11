import unittest

import torch

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego

pystratego = get_pystratego()


def was_battle(board_string, action, current_player):
    coords = pystratego.util.actions_to_abs_coordinates(action, current_player)[0]
    return board_string.replace("@", "").replace(".", "")[coords[1]] != "a"


class LastAttackTest(unittest.TestCase):
    def test_last_attack_method(self):
        num_moves_since_battle = 0
        env = Stratego(num_envs=1, traj_len_per_player=100, barrage=True)
        action_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
        for _ in range(10000):
            action_tensor = env.sample_random_legal_action(action_tensor)
            num_moves_since_battle += 1
            cur_board = env.current_board_strs[0]
            cur_player = env.current_player
            if was_battle(cur_board, action_tensor, cur_player) and not env.current_is_terminal:
                num_moves_since_battle = 0
            env.apply_actions(action_tensor)
            if env.current_num_moves_since_reset == 0:
                num_moves_since_battle = 0
            self.assertTrue(num_moves_since_battle == env.current_num_moves_since_last_attack)

    def test_last_attack_plane(self):
        num_moves_since_battle = 0
        env = Stratego(num_envs=1, traj_len_per_player=100)
        action_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
        for _ in range(10000):
            action_tensor = env.sample_random_legal_action(action_tensor)
            num_moves_since_battle += 1
            if (
                was_battle(env.current_board_strs[0], action_tensor, env.current_player)
                and not env.current_is_terminal
            ):
                num_moves_since_battle = 0
            env.apply_actions(action_tensor)
            if env.current_num_moves_since_reset == 0:
                num_moves_since_battle = 0
            self.assertTrue(
                torch.allclose(
                    torch.tensor(num_moves_since_battle, device="cuda", dtype=torch.float32),
                    (
                        env.conf.max_num_moves_between_attacks
                        * env.current_infostate_tensor[
                            :,
                            env.INFOSTATE_CHANNEL_DESCRIPTION.index(
                                "max_num_moves_between_attacks_frac"
                            ),
                        ]
                    ),
                )
            )

    def test_termination(self):
        env = Stratego(num_envs=100, traj_len_per_player=100)
        action_tensor = torch.zeros(100, device="cuda", dtype=torch.int32)
        for _ in range(10000):
            action_tensor = env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            is_over_battle_limit = (
                env.current_num_moves_since_last_attack
                >= env.conf.max_num_moves_between_attacks + 1
            )
            terminated_since = env.env.get_terminated_since(env.current_step)
            not_already_terminated = terminated_since < 2
            mask = is_over_battle_limit & not_already_terminated
            if mask.any():
                self.assertTrue(env.current_is_terminal[mask].all())
                self.assertTrue(
                    torch.allclose(
                        env.current_reward_pl0[mask],
                        torch.zeros_like(env.current_reward_pl0[mask]),
                    )
                )

    def test_last_attack_cache(self):
        env = Stratego(num_envs=1, traj_len_per_player=100, barrage=True)
        action_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
        num_moves_since_battle_history = []
        for t in range(10000):
            action_tensor = env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            num_moves_since_battle_history.append(env.current_num_moves_since_last_attack.item())
            if t % 100 == 0 and t > 0:
                for t_ in range(99):
                    self.assertTrue(
                        torch.allclose(
                            env.num_moves_since_last_attack(env.current_step - t_),
                            torch.tensor(
                                num_moves_since_battle_history[-1 - t_],
                                device="cuda",
                                dtype=torch.int32,
                            ),
                        )
                    )


if __name__ == "__main__":
    unittest.main()
