import argparse

from pyengine.core.evaluation import evaluate
from pyengine.networks.wrappers import ThresholdModel
from pyengine.utils.loading import load_rl_model, load_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=1000, type=int)
    parser.add_argument("--threshold", default=0.01, type=float)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--opponent_path", required=True)
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    assert 0 <= args.threshold

    model, arrangements = load_rl_model(args.model_path)
    opponent, opponent_arrangements = load_rl_model(args.opponent_path)
    env = load_env(args.model_path, args.num_envs)

    threshold_model = ThresholdModel(model, args.threshold)

    result = (
        evaluate(threshold_model, arrangements, opponent, opponent_arrangements, env).mean().item()
    )
    print(result)
