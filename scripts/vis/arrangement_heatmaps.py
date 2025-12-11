from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import pickle

from pyengine.utils.loading import load_arrangement_model
from pyengine.arrangement.sampling import generate_arrangements
from pyengine.arrangement.utils import to_string, filter_terminal

char_to_name = {
    "C": "SPY",
    "D": "SCOUT",
    "E": "MINER",
    "F": "SERGEANT",
    "G": "LIEUTENANT",
    "H": "CAPTAIN",
    "I": "MAJOR",
    "J": "COLONEL",
    "K": "GENERAL",
    "L": "MARSHAL",
    "M": "FLAG",
    "B": "BOMB",
    "_": "LAKE",
    "A": "EMPTY",
}


def plot_heatmaps_for_each_piece(boards_list, save_dir):
    # Ensure that each board in the list has 40 characters
    assert all(
        len(board) == 40 for board in boards_list
    ), "Each board must be a string of exactly 40 characters."

    # Convert the list of boards (strings) into a numpy array of shape (num_boards, 4, 10)
    boards_array = np.array([list(board) for board in boards_list]).reshape(-1, 4, 10)
    boards_array = np.flip(boards_array, axis=1)  # Flip the board to reverse the perspective
    num_boards = boards_array.shape[0]

    # Get the unique characters from all boards
    unique_pieces = sorted(set(np.ravel(boards_array)))

    # Initialize a dictionary to store frequency counts for each piece
    piece_count_map = {piece: np.zeros((4, 10)) for piece in unique_pieces}

    # Count the frequency of each piece in each position across all boards
    for board in boards_array:
        for row in range(4):
            for col in range(10):
                piece = board[row, col]
                piece_count_map[piece][row, col] += 1

    # Custom labels
    row_labels = [4, 3, 2, 1]  # Descending order for rows
    col_labels = [
        chr(i) for i in range(ord("a"), ord("k"))
    ]  # Columns labeled with lowercase letters

    # Plot heatmap for each piece
    for piece, count_matrix in piece_count_map.items():
        proportion_matrix = count_matrix / num_boards  # Calculate proportion
        piece_name = char_to_name.get(piece, piece)  # Get piece name from dictionary

        # Use the 'RdYlGn_r' colormap for better contrast and visibility
        fig, ax = plt.subplots(figsize=(10, 10))  # Making the board square
        ax.imshow(proportion_matrix, cmap="RdYlGn_r", aspect="equal", interpolation="nearest")

        # Only keep the square borders
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(10) - 0.5, minor=True)
        ax.set_yticks(np.arange(4) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", size=0)

        # Use black text to ensure visibility
        for i in range(4):
            for j in range(10):
                percentage = f"{proportion_matrix[i, j] * 100:.1f}%"
                ax.text(j, i, percentage, ha="center", va="center", color="black", fontsize=12)

        # Labels and title (removing colorbar)
        ax.set_title(f"Heatmap for {piece_name}")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_xticks(np.arange(10))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(row_labels)

        plt.savefig(f"{save_dir}/{piece_name}_heatmap.png")
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--init_model_path", type=str)
    parser.add_argument("--arrangements_path", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    # Ensure that either init_model_path or arrangements_path is provided, but not both
    assert (args.init_model_path is None) != (
        args.arrangements_path is None
    ), "Either init_model_path or arrangements_path must be provided, but not both."

    if args.init_model_path:
        init_model = load_arrangement_model(args.init_model_path)
        tensor_arrangements, *_ = generate_arrangements(10000, init_model)
        arrangements = filter_terminal(to_string(tensor_arrangements))
    else:
        with open(args.arrangements_path, "rb") as f:
            arrangements = pickle.load(f)
        arrangements = arrangements[0] + arrangements[1]
    forced_handedness = []
    for a in arrangements:
        if "M" in (a[5:10] + a[15:20] + a[25:30] + a[35:40]):
            a_ = a[0:10][::-1] + a[10:20][::-1] + a[20:30][::-1] + a[30:40][::-1]
            forced_handedness.append(a_)
        else:
            forced_handedness.append(a)
    plot_heatmaps_for_each_piece(forced_handedness, args.save_dir)
