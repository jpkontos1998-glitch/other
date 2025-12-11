import argparse
from pyengine.utils.init_helpers import is_valid_line, piece_counters, COUNTERS

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="classic", choices=["classic", "barrage"])
parser.add_argument("--max-num-boards", default=128, type=int)


def flip_cols(s):
    assert len(s) == 40
    return s[:10][::-1] + s[19:9:-1] + s[29:19:-1] + s[39:29:-1]


if __name__ == "__main__":
    args = parser.parse_args()
    lines = open(f"{args.variant}_human_inits.dat").readlines()
    max_num_boards = args.max_num_boards if args.max_num_boards > 0 else 10**15
    lines = [line.strip()[::-1] for line in lines if is_valid_line(line)][: args.max_num_boards]
    lines += [flip_cols(line) for line in lines]
    lines = set(lines)

    print(f"const std::array<const char*, {len(lines)}> JB_INIT_BOARDS_{args.variant.upper()} = {{")
    for line in lines:
        assert piece_counters(line) == COUNTERS[args.variant]
    print(",\n".join([f'    "{line}"' for line in lines]))
    print("};")
