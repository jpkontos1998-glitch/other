import argparse

from pyengine.core.evaluation import evaluate
from pyengine.networks.wrappers import ClampModel
from pyengine.utils.loading import load_rl_model, load_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=1000, type=int)
    parser.add_argument("--clamp_val", default=1e-5, type=float)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    assert 0 <= args.clamp_val <= 1e-2

    model, arrangements = load_rl_model(args.model_path)
    env = load_env(args.model_path, args.num_envs)

    clamp_model = ClampModel(model, args.clamp_val)

    result = evaluate(clamp_model, arrangements, model, arrangements, env).mean().item()
    print(result)
