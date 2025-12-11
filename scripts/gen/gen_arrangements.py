import argparse
import pickle
import re

from pyengine.arrangement.sampling import generate_arrangements
from pyengine.arrangement.utils import to_string, filter_terminal
from pyengine.utils.loading import load_arrangement_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", required=True)
    parser.add_argument("--num_arrangements", default=1000, type=int)
    parser.add_argument("--save_dir")
    args = parser.parse_args()

    model = load_arrangement_model(args.init_model)
    tensor_arrangements, *_ = generate_arrangements(args.num_arrangements, model)
    arrangements = filter_terminal(to_string(tensor_arrangements))
    model_number = re.search(r"init_model(.*)\.pth*", args.init_model).group(1)
    ema = "" if "pthm" not in args.init_model else "ema_"
    with open(f"{args.save_dir}/{ema}arrangements{model_number}.pkl", "wb") as f:
        pickle.dump([arrangements, arrangements], f)
