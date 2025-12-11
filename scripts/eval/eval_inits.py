import argparse
import pickle


from pyengine.core.evaluation import evaluate
from pyengine.utils.loading import (
    get_train_info,
    load_rl_model,
    log_dir_from_fn,
    load_env,
)
from pyengine import utils

pystratego = utils.get_pystratego()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=100, type=int)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_path2", type=str, default=None)
    parser.add_argument("--arrangements_path", type=str)
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()
    train_info = get_train_info(log_dir_from_fn(args.model_path))
    env = load_env(args.model_path, num_envs=args.num_envs)
    model, arrangements = load_rl_model(args.model_path)
    with open(args.arrangements_path, "rb") as f:
        other_arrangements = pickle.load(f)
    if args.model_path2 is not None:
        model2, arrangements = load_rl_model(args.model_path2)
    else:
        model2 = model
    print("Performance:", evaluate(model, other_arrangements, model2, arrangements, env).mean())
