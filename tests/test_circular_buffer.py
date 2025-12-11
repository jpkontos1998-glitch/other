import unittest

import torch

from pyengine.core.buffer import CircularBuffer


class CircularBufferTest(unittest.TestCase):
    def test_terminal_case1(self):
        """
        test case 1: (not term, term), (term, term), new game

        rollout terminal:         0, 1, 1, 1, 0, 0
        expected output terminal: 1, 1, 1, 1, 0, 0

        rollout reward:          0, 1, -1, 1, 0, 0
        expected output reward: -1, 1, -1, 1, 0, 0
        """

        traj_len = 3
        buffer = CircularBuffer(
            num_envs=1,
            traj_len=traj_len,
            train_every_per_player=traj_len,
            use_cat_vf=False,
            device="cuda",
        )
        # t=0: first player move, not term yet
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([0], device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )

        # t=1: 2nd player move, game ends
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([1], device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )

        # t=2: 1st player move, game ends
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([-1], device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )

        # t=3: 2nd player move, game ends
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([1], device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )

        # t=4: 1st player move, new game
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([0], device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )

        # input terminal: 0, 1, 1, 1, 0
        # expected output terminal: 1, 1, 1, 1, 0
        terms = buffer.is_terminating_action.squeeze(1).cpu()
        expected_terms = torch.tensor([1, 1, 1, 1, 0, 0]).float()
        assert torch.equal(terms, expected_terms)

        # input reward: 0, 1, -1, 1, 0
        # expected output reward: -1, 1, -1, 1, 0
        rewards = buffer.terminal_rewards.squeeze(1).cpu()
        expected_rewards = torch.tensor([-1, 1, -1, 1, 0, 0]).float()
        assert torch.equal(rewards, expected_rewards)

    def test_terminal_case2(self):
        """
        test case 1: (not term, not term), (term, term), new game

        rollout terminal:         0, 0, 1, 1, 0
        expected output terminal: 0, 1, 1, 1, 0

        rollout reward:         0,  0, 1, -1, 0
        expected output reward: 0, -1, 1, -1, 0
        """
        traj_len = 3
        buffer = CircularBuffer(
            num_envs=1,
            traj_len=traj_len,
            train_every_per_player=traj_len,
            use_cat_vf=False,
            device="cuda",
        )
        # t=0: first player move, not term yet
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([0], device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )

        # t=1: 2nd player move, game ends
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([0], device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )

        # t=2: 1st player move, game ends
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([1], device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )

        # t=3: 2nd player move, game ends
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([-1], device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )

        # t=4: 1st player move, new game
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([0], device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )

        # input terminal: 0, 0, 1, 1, 0, 0
        # expected output terminal: 0, 1, 1, 1, 0, 0
        terms = buffer.is_terminating_action.squeeze(1).cpu()
        expected_terms = torch.tensor([0, 1, 1, 1, 0, 0]).float()
        assert torch.equal(terms, expected_terms)

        # input reward: 0, 0, -1, 1, 0, 0
        # expected output reward: 0, -1, 1, -1, 0, 0
        rewards = buffer.terminal_rewards.squeeze(1).cpu()
        expected_rewards = torch.tensor([0, -1, 1, -1, 0, 0]).float()
        assert torch.equal(rewards, expected_rewards)

    def test_1step_return(self):
        """
        terminals: 0, 0, 1, 1, 1, 0
        rewards  : 0, 0, -1, 1, -1, 0
        value    : 0, 0, -0.9, 1, -1, 0

        expected return  :  -0.9, 1.0, -1, 1, -1, 0
        """
        traj_len = 4
        buffer = CircularBuffer(
            num_envs=1,
            traj_len=traj_len,
            train_every_per_player=traj_len,
            use_cat_vf=False,
            device="cuda",
        )
        # t=0, 1: not terminated yet
        for _ in range(2):
            buffer.add_pre_act(
                step=0,
                num_moves=torch.tensor([0], device="cuda"),
                legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
                is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
            )
            buffer.add_post_act(
                action=torch.tensor([1]),
                value=torch.tensor([0]).float(),
                log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
                reward=torch.tensor([0], device="cuda"),
                is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
            )

        # t=2, 1st player is about to lose, its value prediction becomes -0.9
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([-0.9]).float(),
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([0], device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )

        # t=3: 2nd player move, game ends
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([1]).float(),  # assume value prediction is correct
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([1], device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )

        # t=4: 1st player move, game ends
        buffer.add_pre_act(
            step=0,
            num_moves=torch.tensor([0], device="cuda"),
            legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )
        buffer.add_post_act(
            action=torch.tensor([1]),
            value=torch.tensor([-1]).float(),  # assume value prediction is correct
            log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
            reward=torch.tensor([-1], device="cuda"),
            is_terminal=torch.tensor([1], device="cuda", dtype=torch.bool),
        )

        # t=5,6,7: 1st, 2nd, 1st player move, new game
        for i in range(3):
            buffer.add_pre_act(
                step=0,
                num_moves=torch.tensor([0], device="cuda"),
                legal_action_mask=torch.zeros((1, 1800), dtype=torch.bool, device="cuda"),
                is_terminal=torch.tensor([int(i == 0)], device="cuda", dtype=torch.bool),
            )
            buffer.add_post_act(
                action=torch.tensor([1]),
                value=torch.tensor([0]).float(),
                log_prob=torch.zeros((1, 1800), dtype=torch.float, device="cuda"),
                reward=torch.tensor([0], device="cuda"),
                is_terminal=torch.tensor([0], device="cuda", dtype=torch.bool),
            )

        # terminals: 0, 0, 1, 1, 1, 0
        # rewards  : 0, 0, -1, 1, -1, 0
        # value    : 0, 0, -0.9, 1, -1, 0
        expected_return = torch.tensor([-0.9, 1.0, -1, 1, -1, 0, 0, 0]).float()
        buffer.process_data(td_lambda=0.0, gae_lambda=0.0)  # 1-step return
        returns = buffer.returns.squeeze(1).cpu()
        assert torch.equal(returns, expected_return)


if __name__ == "__main__":
    unittest.main()
