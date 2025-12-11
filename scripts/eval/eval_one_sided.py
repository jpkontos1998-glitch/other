import argparse

from pyengine.core.evaluation import evaluate_one_sided
from pyengine.utils.loading import get_train_info, log_dir_from_fn, load_rl_model, load_env
from pyengine import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=100, type=int)
    parser.add_argument("--model_path1", required=True)
    parser.add_argument("--model_path2", required=True)
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    train_infos1 = get_train_info(log_dir_from_fn(args.model_path1))
    train_infos2 = get_train_info(log_dir_from_fn(args.model_path2))
    assert train_infos1["full_info"] == train_infos2["full_info"]
    assert train_infos1["barrage"] == train_infos2["barrage"]
    env = load_env(args.model_path1, args.num_envs)
    model1, arrangements1 = load_rl_model(args.model_path1)
    model2, arrangements2 = load_rl_model(args.model_path2)
    with utils.eval_mode(model1, model2):
        perf = (
            evaluate_one_sided(model1, arrangements1[0], model2, arrangements2[1], env)
            .mean()
            .item()
        )
    print(round(perf, 2))
