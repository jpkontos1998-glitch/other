import unittest

import torch

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego, set_seed_everywhere

pystratego = get_pystratego()


class DMPlaneTest(unittest.TestCase):
    def test_dm_planes(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        plane_idx = []
        for i, description in enumerate(env.env.INFOSTATE_CHANNEL_DESCRIPTION):
            if description[:2] == "dm":
                plane_idx.append(i)
        plane_idx = torch.tensor(plane_idx, device="cuda")
        my_dm_planes = torch.zeros((1, plane_idx.shape[0], 10, 10), device="cuda")
        set_seed_everywhere(0)
        for i in range(10000):
            new_plane = torch.zeros((1, 1, 10, 10), device="cuda")
            a = env.sample_random_legal_action()
            coordinates = pystratego.util.actions_to_abs_coordinates(a, env.current_player)[0]
            # Place 1 at destination square
            new_plane[0, 0, coordinates[1] // 10, coordinates[1] % 10] = 1
            # Note this is in relative coordinates so we need to flip if player 1 is acting
            piece_types = env.current_piece_type_onehot
            if env.current_player == 1:
                piece_types = piece_types.flip((1, 2))
            piece_to_move = (
                piece_types[0, coordinates[0] // 10, coordinates[0] % 10].int().argmax() + 1
            )
            piece_at_dest = (
                piece_types[0, coordinates[1] // 10, coordinates[1] % 10].int().argmax() + 1
            )

            if piece_at_dest == 14:  # Place -1 at origin if no attack
                new_plane[0, 0, coordinates[0] // 10, coordinates[0] % 10] = -1
            else:  # Place scaled piece value otherwise
                new_plane[0, 0, coordinates[0] // 10, coordinates[0] % 10] = -(
                    2 + piece_to_move / 12
                )

            env.apply_actions(a)
            # Toss out the oldest plane
            my_dm_planes = torch.cat((my_dm_planes, new_plane), dim=1)[:, 1:]
            while env.current_is_terminal:
                a = env.sample_random_legal_action()
                env.apply_actions(a)
                my_dm_planes = torch.zeros_like(my_dm_planes)

            dm_planes = env.current_infostate_tensor[:, plane_idx]
            if env.current_player == 1:
                my_dm_planes_tmp = my_dm_planes.flip((2, 3))
            else:
                my_dm_planes_tmp = my_dm_planes
            self.assertTrue(torch.allclose(dm_planes, my_dm_planes_tmp))

    def test_dm_planes_with_reset(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        plane_idx = []
        for i, description in enumerate(env.env.INFOSTATE_CHANNEL_DESCRIPTION):
            if description[:2] == "dm":
                plane_idx.append(i)
        plane_idx = torch.tensor(plane_idx, device="cuda")
        my_dm_planes = torch.zeros((1, plane_idx.shape[0], 10, 10), device="cuda")
        saved_states_ls = []
        saved_planes_ls = []
        my_dm_planes_ls = []
        saved_board_str = []
        set_seed_everywhere(0)
        time_steps = 1  # 250  #  10000
        for i in range(time_steps):
            new_plane = torch.zeros((1, 1, 10, 10), device="cuda")
            a = env.sample_random_legal_action()
            coordinates = pystratego.util.actions_to_abs_coordinates(a, env.current_player)[0]
            # Place 1 at destination square
            new_plane[0, 0, coordinates[1] // 10, coordinates[1] % 10] = 1
            # Note this is in relative coordinates so we need to flip if player 1 is acting
            piece_types = env.current_piece_type_onehot
            if env.current_player == 1:
                piece_types = piece_types.flip((1, 2))
            # Note the add 1 on the next two lines
            piece_to_move = (
                piece_types[0, coordinates[0] // 10, coordinates[0] % 10].int().argmax() + 1
            )
            piece_at_dest = (
                piece_types[0, coordinates[1] // 10, coordinates[1] % 10].int().argmax() + 1
            )

            if piece_at_dest == 14:  # Place -1 at origin if no attack
                new_plane[0, 0, coordinates[0] // 10, coordinates[0] % 10] = -1
            else:  # Place scaled piece value otherwise
                new_plane[0, 0, coordinates[0] // 10, coordinates[0] % 10] = -(
                    2 + piece_to_move / 12
                )

            env.apply_actions(a)
            # Toss out the oldest plane
            my_dm_planes = torch.cat((my_dm_planes, new_plane), dim=1)[:, 1:]
            while env.current_is_terminal:
                a = env.sample_random_legal_action()
                env.apply_actions(a)
                my_dm_planes = torch.zeros_like(my_dm_planes)

            dm_planes = env.current_infostate_tensor[:, plane_idx]
            if env.current_player == 1:
                my_dm_planes_tmp = my_dm_planes.flip((2, 3))
            else:
                my_dm_planes_tmp = my_dm_planes

            saved_board_str.append(env.current_board_strs[0])
            saved_states_ls.append(env.current_state)
            saved_planes_ls.append(env.current_infostate_tensor.clone())
            my_dm_planes_ls.append(my_dm_planes_tmp.clone())

        # Test that things work on reset states
        for t, (state, saved_planes, my_dm_planes, saved_board) in enumerate(
            zip(saved_states_ls, saved_planes_ls, my_dm_planes_ls, saved_board_str)
        ):
            env.change_reset_behavior_to_env_state(state)
            env.reset()
            dm_planes = env.current_infostate_tensor[:, plane_idx]
            self.assertTrue(saved_board, env.current_board_strs[0])

            if not torch.allclose(env.current_infostate_tensor, saved_planes):
                print(f"=== mismatch between RECOMPUTED AND HISTORICAL planes at time {t}")
                for ch in range(env.env.NUM_INFOSTATE_CHANNELS):
                    if not torch.allclose(env.current_infostate_tensor[0, ch], saved_planes[0, ch]):
                        print(
                            f"Mismatching plane is {ch} (`{env.env.INFOSTATE_CHANNEL_DESCRIPTION[ch]}`)"
                        )
                        print("PLANE WAS:")
                        print(saved_planes[0, ch])
                        print("... BUT AFTER RESETTING THE STATE IT IS")
                        print(env.current_infostate_tensor[0, ch])
            if not torch.allclose(dm_planes, my_dm_planes):
                print(f"=== mismatch of DEEPMIND planes at time {t}")
                for ch in range(
                    env.env.NUM_INFOSTATE_CHANNELS - plane_idx.shape[0],
                    env.env.NUM_INFOSTATE_CHANNELS,
                ):
                    if not torch.allclose(
                        env.current_infostate_tensor[0, ch],
                        my_dm_planes[0, ch - (env.env.NUM_INFOSTATE_CHANNELS - plane_idx.shape[0])],
                    ):
                        print(
                            f"Mismatching plane is {ch} (`{env.env.INFOSTATE_CHANNEL_DESCRIPTION[ch]}`)"
                        )
                        print("FOUND:")
                        print(env.current_infostate_tensor[0, ch])
                        print("EXPECTED")
                        print(
                            my_dm_planes[
                                0, ch - (env.env.NUM_INFOSTATE_CHANNELS - plane_idx.shape[0])
                            ]
                        )

            self.assertTrue(torch.allclose(env.current_infostate_tensor, saved_planes))
            self.assertTrue(torch.allclose(dm_planes, my_dm_planes))


if __name__ == "__main__":
    unittest.main()
