import argparse

from pyengine.core.evaluation import evaluate
from pyengine.networks.wrappers import ArgmaxModel
from pyengine.utils.loading import load_rl_model, load_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=1000, type=int)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    model, arrangements = load_rl_model(args.model_path)
    env = load_env(args.model_path, args.num_envs)

    threshold_model = ArgmaxModel(model)

    result = evaluate(threshold_model, arrangements, model, arrangements, env).mean().item()
    print(result)
