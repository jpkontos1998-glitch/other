import unittest

import torch

from pyengine.core.env import Stratego
from pyengine.belief.uniform import uniform_belief
from pyengine.belief.masking import create_piece_count_mask, create_movement_mask
from pyengine import utils


pystratego = utils.get_pystratego()


def check_valid_counts(
    env_state: pystratego.EnvState,
    samples: torch.Tensor,
) -> torch.Tensor:
    # Check number of hidden pieces of each type matches the counters
    boards = env_state.boards
    expected_counts = boards[:, 1612:1624] if env_state.to_play == 0 else boards[:, 1600:1612]
    num_placed = samples.sum(1)[:, :12].reshape(-1, 12)
    correct_counts = (num_placed == expected_counts).all(-1)
    return correct_counts


def check_valid_immovable(
    samples: torch.Tensor,
    unknown_piece_has_moved: torch.Tensor,
) -> torch.Tensor:
    # Check immovable pieces (BOMB, FLAG) are not assigned to unknown pieces that have moved
    has_moved = unknown_piece_has_moved
    has_bomb_or_flag = samples[:, :, 10:12].reshape(samples.shape[0], samples.shape[1], 2).sum(-1)
    has_bomb_or_flag_in_moved = (has_bomb_or_flag * has_moved).any(-1)
    return ~has_bomb_or_flag_in_moved


def count_remaining_not_moved(unknown_piece_has_moved, unknown_piece_count):
    """
    Computes the number of pieces that have not moved for each (n, k) where k' >= k.

    Parameters:
    - unknown_piece_has_moved (torch.Tensor): A boolean tensor of shape (N, K)
                                              indicating if each piece has moved.
    - unknown_piece_count (torch.Tensor): An integer tensor of shape (N,)
                                          indicating the number of unknown pieces for each N.

    Returns:
    - torch.Tensor: A tensor of shape (N, K) where each element (n, k) is the number of pieces
                    k' >= k that have not moved.
    """
    N, K = unknown_piece_has_moved.shape

    # Create a mask to ignore out-of-bound pieces
    range_tensor = torch.arange(K, device="cuda").unsqueeze(0).expand(N, -1)
    valid_piece_mask = range_tensor < unknown_piece_count.unsqueeze(1)

    # Invert the masked tensor to find pieces that have not moved
    unknown_piece_has_not_moved = (~unknown_piece_has_moved) & valid_piece_mask

    # Compute the cumulative sum in the reverse direction to count k' >= k
    remaining_not_moved_count = torch.cumsum(
        unknown_piece_has_not_moved.flip(dims=[1]), dim=1
    ).flip(dims=[1])

    return remaining_not_moved_count


class BeliefTest(unittest.TestCase):
    def test_counts(self):
        utils.set_seed_everywhere(0)
        num_envs = 10
        for barrage in [True, False]:
            env = Stratego(
                num_envs=num_envs, traj_len_per_player=100, full_info=False, barrage=barrage
            )
            action_tensor = torch.zeros(num_envs, device="cuda", dtype=torch.int32)
            for _ in range(100):
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)
                samples = torch.cat([
                    uniform_belief.generate(
                    n_sample=1,
                    unknown_piece_position_onehot=env.current_unknown_piece_position_onehot[i],
                    unknown_piece_has_moved=env.current_unknown_piece_has_moved[i],
                    unknown_piece_counts=env.current_unknown_piece_counts[i],
                    )
                    for i in range(num_envs)
                ])
                is_valid = check_valid_counts(env.current_state, samples)
                self.assertTrue(is_valid.all())

    def test_immovable(self):
        utils.set_seed_everywhere(0)
        num_envs = 10
        for barrage in [True, False]:
            env = Stratego(
                num_envs=num_envs, traj_len_per_player=100, full_info=False, barrage=barrage
            )
            action_tensor = torch.zeros(num_envs, device="cuda", dtype=torch.int32)
            for _ in range(100):
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)
                samples = torch.cat([
                    uniform_belief.generate(
                    n_sample=1,
                    unknown_piece_position_onehot=env.current_unknown_piece_position_onehot[i],
                    unknown_piece_has_moved=env.current_unknown_piece_has_moved[i],
                    unknown_piece_counts=env.current_unknown_piece_counts[i],
                    )
                    for i in range(num_envs)
                ])
                is_valid = check_valid_immovable(samples, env.current_unknown_piece_has_moved)
                self.assertTrue(is_valid.all())

    def test_consistency(self):
        utils.set_seed_everywhere(0)
        num_envs = 10
        for barrage in [True, False]:
            env = Stratego(
                num_envs=num_envs, traj_len_per_player=100, full_info=False, barrage=barrage
            )
            action_tensor = torch.zeros(num_envs, device="cuda", dtype=torch.int32)
            for _ in range(100):
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)
                mask = create_piece_count_mask(
                    env.current_unknown_piece_counts, env.current_unknown_piece_type_onehot
                ) & create_movement_mask(
                    env.current_unknown_piece_counts,
                    env.current_unknown_piece_type_onehot,
                    env.current_unknown_piece_has_moved,
                )
                piecetypes = env.current_unknown_piece_type_onehot.int()
                for i in range(num_envs):
                    for k in range(mask.shape[1]):
                        if piecetypes[i, k].any():
                            self.assertTrue(mask[i, k, piecetypes[i, k].argmax(dim=-1)])


if __name__ == "__main__":
    unittest.main()
