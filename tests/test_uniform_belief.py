import unittest

import torch

from pyengine.core.env import Stratego
from pyengine.belief.uniform import marginalized_uniform_belief, uniform_belief
from pyengine import utils


pystratego = utils.get_pystratego()


class UniformBeliefTest(unittest.TestCase):
    def test_consistency(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
            barrage=True,
        )
        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for i in range(100):
            if not env.current_is_terminal:
                uniform = uniform_belief(
                    env.current_unknown_piece_type_onehot,
                    env.current_unknown_piece_has_moved,
                    env.current_unknown_piece_counts,
                )
                marginalized_uniform = marginalized_uniform_belief(
                    env.current_infostate_tensor,
                    env.current_unknown_piece_position_onehot,
                )
                # Uniform and marginalized uniform should agree on the first piece dimension
                self.assertTrue(torch.allclose(uniform[:, 0].exp(), marginalized_uniform[:, 0].exp()))
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

    def test_marginal(self):
        for barrage in [True, False]:
            env = Stratego(
                num_envs=1,
                traj_len_per_player=100,
                full_info=False,
                barrage=barrage,
            )
            action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
            for i in range(100):
                if not env.current_is_terminal:
                    generated = uniform_belief.generate(
                        n_sample=10_000,
                        unknown_piece_position_onehot=env.current_unknown_piece_position_onehot[0],
                        unknown_piece_has_moved=env.current_unknown_piece_has_moved[0],
                        unknown_piece_counts=env.current_unknown_piece_counts[0],
                    )
                    marginalized_uniform = marginalized_uniform_belief.generate(
                        n_sample=10_000,
                        unknown_piece_position_onehot=env.current_unknown_piece_position_onehot[0],
                        unknown_piece_has_moved=env.current_unknown_piece_has_moved[0],
                        unknown_piece_counts=env.current_unknown_piece_counts[0],
                        infostate_tensor=env.current_infostate_tensor[0],
                    )
                    uniform = marginalized_uniform_belief(
                        env.current_infostate_tensor,
                        env.current_unknown_piece_position_onehot,
                    )[0].exp()
                    uniform[env.current_unknown_piece_counts.sum() :] = 0
                    empirical = generated.float().mean(dim=0)
                    self.assertTrue(torch.allclose(uniform, empirical, atol=5e-2))
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)


if __name__ == "__main__":
    unittest.main()
