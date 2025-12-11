from typing import Callable, Union
import torch

from pyengine.core.env import Stratego
from pyengine.core.search import SearchBot
from pyengine.utils.loading import load_rl_model
from pyengine import utils

pystratego = utils.get_pystratego()

Model = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]
ActionTaker = Callable[[torch.Tensor], torch.Tensor]
Agent = Union[Model, ActionTaker, SearchBot]


def extract_action(
    agent: Agent,
    env: Stratego,
) -> torch.Tensor:
    if isinstance(agent, SearchBot):
        (even_states, odd_states) = env.snapshot_env_history(env.current_step, 0)
        env_state = even_states if env.current_player == 0 else odd_states
        actions = agent(
            env_state,
            env.current_infostate_tensor,
            env.current_piece_ids,
            env.current_legal_action_mask,
            env.current_num_moves,
            env.current_unknown_piece_position_onehot,
            env.current_unknown_piece_counts,
            env.current_unknown_piece_has_moved,
        )
    elif utils.extended_isinstance(agent, Model):
        actions = agent(
            infostate_tensor=env.current_infostate_tensor,
            piece_ids=env.current_piece_ids,
            legal_action_mask=env.current_legal_action_mask,
            num_moves=env.current_num_moves,
            acting_player=env.current_acting_player,
        )["action"]
    elif utils.extended_isinstance(agent, ActionTaker):
        actions = agent(env.current_legal_action_mask)
    else:
        raise ValueError(f"Unknown agent type {type(agent)}")
    return actions


class EvaluationManager:
    def __init__(self, env: Stratego):
        self.env = env
        self.default_arrangements = env.conf.initial_arrangements
        self.eval_against = []

    def add_checkpoint(
        self,
        model_path: str,
    ) -> None:
        network, arr = load_rl_model(model_path)
        self.eval_against.append((network, arr))

    def evaluate(self, model: Agent, model_arrangements) -> dict[str, float]:
        performance = {}
        for i, (opp, opp_arr) in enumerate(self.eval_against):
            performance[f"vs_model{i}"] = (
                evaluate(model, model_arrangements, opp, opp_arr, self.env).mean().item()
            )
        return performance


def evaluate_one_sided(
    pl0_agent: Agent,
    pl0_arrangements: list[str],
    pl1_agent: Agent,
    pl1_arrangements: list[str],
    env: Stratego,
) -> torch.Tensor:
    """Returns a tensor, containing the reward of pl0 for each of the envs in the Stratego object."""
    env.change_reset_behavior_to_random_initial_arrangement([pl0_arrangements, pl1_arrangements])
    env.reset()
    already_terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    outcomes_pl0 = torch.zeros(env.num_envs, device=env.device)
    action_tensor = torch.empty(env.num_envs, device=env.device, dtype=torch.int32)
    for _ in range(env.conf.max_num_moves):
        env.sample_random_legal_action(action_tensor)
        acting_agent = pl0_agent if env.current_player == 0 else pl1_agent
        action_tensor = extract_action(
            acting_agent,
            env,
        )
        env.apply_actions(action_tensor)
        is_terminal = env.current_is_terminal
        rewards = env.current_reward_pl0
        outcomes_pl0 += torch.logical_not(already_terminated) * is_terminal * rewards
        already_terminated = torch.logical_or(already_terminated, is_terminal)
        if already_terminated.sum() == env.num_envs:
            break
    return outcomes_pl0


def evaluate(
    agent_A: Agent,
    arrangements_A: tuple[list[str], list[str]],
    agent_B: Agent,
    arrangements_B: tuple[list[str], list[str]],
    env: Stratego,
) -> torch.Tensor:
    """
    Returns a tensor of size 2 * env.num_envs.
    The first half contains the reward of agent_A when it plays player 0,
    the second half the reward of agent_A when it plays player 1.
    Play from same set of starting state to reduce variance
    """
    pl0_evals = evaluate_one_sided(agent_A, arrangements_A[0], agent_B, arrangements_B[1], env)
    pl1_evals = -evaluate_one_sided(agent_B, arrangements_B[0], agent_A, arrangements_A[1], env)
    return torch.cat([pl0_evals, pl1_evals], dim=0)
