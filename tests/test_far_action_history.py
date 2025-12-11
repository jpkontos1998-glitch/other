import unittest
import torch

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego

pystratego = get_pystratego()

OLD_NUM_BOARD_STATE_CHANNELS = 42


class FarActionHistoryTest(unittest.TestCase):
    def test_run(self):
        move_memory = 10
        env = Stratego(num_envs=1, traj_len_per_player=50, move_memory=move_memory)
        num_moves_frac_channel = env.INFOSTATE_CHANNEL_DESCRIPTION.index("max_num_moves_frac")
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

        actions = []
        for _ in range(env.env.buf_size * 40 + 10):
            env.sample_random_legal_action(action_tensor)
            actions += [action_tensor.item() if not env.current_is_terminal else None]
            env.apply_actions(action_tensor)

        print("---")
        for t in range(env.current_step - env.env.buf_size + 1 + move_memory, env.current_step + 1):
            print(
                f"t={t}  terminated_since={env.env.get_terminated_since(t).item()}  num_moves={env.num_moves(t).item():4d}  played={env.env.get_played_actions(t).item() if t != env.current_step else None}"
            )
        print("---")
        len(actions)
        print(f"Len of actions: {len(actions)}")

        for t in range(env.current_step - env.env.buf_size + 1 + move_memory, env.current_step):
            if actions[t] is not None:
                self.assertEqual(env.played_actions(t).item(), actions[t])

        for t in range(env.current_step - env.env.buf_size + 1 + move_memory, env.current_step + 1):
            self.assertTrue(
                torch.allclose(
                    env.infostate_tensor(t)[:, num_moves_frac_channel]
                    .min(-1)
                    .values.min(-1)
                    .values,
                    env.num_moves(t) / env.conf.max_num_moves,
                )
            )
            self.assertTrue(
                torch.allclose(
                    env.infostate_tensor(t)[:, num_moves_frac_channel]
                    .max(-1)
                    .values.max(-1)
                    .values,
                    env.num_moves(t) / env.conf.max_num_moves,
                )
            )

            print(f"Move memory: {env.conf.move_memory}")
            for channel_id in range(
                pystratego.NUM_BOARD_STATE_CHANNELS,
                pystratego.NUM_BOARD_STATE_CHANNELS + env.conf.move_memory,
            ):
                channel = env.infostate_tensor(t)[0, channel_id, :, :]
                time_delta = -(
                    channel_id - pystratego.NUM_BOARD_STATE_CHANNELS - env.conf.move_memory
                )
                if time_delta > env.num_moves(t).item():
                    self.assertEqual(channel.count_nonzero(), 0)
                else:
                    t_past = t - time_delta

                    if actions[t_past] is not None:
                        self.assertEqual(channel.max(), 1.0)
                        self.assertEqual(channel.min(), -1.0)
                        self.assertEqual(channel.count_nonzero(), 2)

                        from_cell = channel.view(100).argmin().item()
                        to_cell = channel.view(100).argmax().item()

                        # The action in the channel is from the point of view of the observer.
                        # The acting player though might be different, and since the
                        # action numbers are always from the point of view of the acting players,
                        # we selectively flip the coordinates.
                        if t_past % 2 != t % 2:
                            from_cell = 99 - from_cell
                            to_cell = 99 - to_cell

                        if from_cell % 10 == to_cell % 10:
                            # Vertical movement
                            new_coord = to_cell // 10
                            if new_coord > from_cell // 10:
                                new_coord -= 1
                        else:
                            new_coord = to_cell % 10
                            if new_coord > from_cell % 10:
                                new_coord -= 1
                            new_coord += 9

                        action = from_cell + 100 * new_coord
                        if actions[t_past] != action:
                            print(
                                from_cell,
                                "->",
                                to_cell,
                                "found",
                                actions[t_past],
                                "expected",
                                action,
                            )
                            self.fail()

                    else:
                        self.assertEqual(channel.max(), 0.0)
                        self.assertEqual(channel.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
