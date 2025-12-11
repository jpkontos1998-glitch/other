import argparse
import pprint
import pickle
import os

from pyengine.core.evaluation import evaluate
from pyengine.utils.loading import get_train_info, log_dir_from_fn, load_rl_model, load_env
from pyengine import utils
from pyengine.utils.loading import get_checkpoint_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=100, type=int)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    all_models = [fn for fn in utils.get_all_files(args.log_dir, "pthw") if "init_model" not in fn]
    all_models = sorted(all_models, key=get_checkpoint_step)
    train_infos = [get_train_info(log_dir_from_fn(fn)) for fn in all_models]

    pprint.pprint(train_infos[0])
    env = load_env(all_models[0], args.num_envs)

    scores = []
    opponent, opponent_arrangements = load_rl_model(args.opponent)
    for model_file in all_models[:3]:
        model, model_arrangements = load_rl_model(model_file)
        score = (
            evaluate(model, model_arrangements, opponent, opponent_arrangements, env).mean().item()
        )
        model_step = get_checkpoint_step(model_file)
        scores.append((model_step, score))

    fig, ax = utils.generate_grid(1, 1, figsize=7)
    xs = [i[0] for i in scores]
    ys = [i[1] for i in scores]
    ax.plot(xs, ys)
    ax.set_ylabel("score")
    ax.set_xlabel("train step")
    opponent_name = "_".join(args.opponent.split("/")[-2:])
    save_path = os.path.join(args.log_dir, f"vs_{opponent_name}.png")
    print(f"saving to {save_path}")
    fig.tight_layout()
    fig.savefig(save_path)
    pickle.dump(scores, open(save_path.replace(".png", ".pkl"), "wb"))
