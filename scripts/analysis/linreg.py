import numpy as np
import argparse
from numpy.linalg import lstsq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect games using an RL model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to save the results")

    args = parser.parse_args()

    all_feat = np.load(f"{args.data_path}/all_feat.npy")
    all_values = np.load(f"{args.data_path}/all_values.npy")
    new_all_feat = np.zeros((all_feat.shape[0], all_feat.shape[1] // 2 - 4))
    for i in range(all_feat.shape[1] // 2 - 4):
        new_all_feat[:, i] = all_feat[:, 4 + 2 * i] - all_feat[:, 4 + 2 * i + 1]
    x = lstsq(
        new_all_feat,
        all_values,
    )[0]
    res = all_values - np.dot(new_all_feat, x)
    res_abs = np.abs(res)
    print("Hidden and visible:")
    print(x.T)
    print(f"Mean absolute error: {res_abs.mean()}")
    print(f"Max absolute error: {res_abs.max()}")
    print(f"Min absolute error: {res_abs.min()}")
    print(f"Std absolute error: {res_abs.std()}")
    from sklearn import tree
    reg = tree.DecisionTreeRegressor()
    reg = reg.fit(new_all_feat, all_values)
    y_hat = reg.predict(new_all_feat)
    res = all_values - y_hat
    res_abs = np.abs(res)
    print("Decision tree:")
    print(f"Mean absolute error: {res_abs.mean()}")
    print(f"Max absolute error: {res_abs.max()}")
    print(f"Min absolute error: {res_abs.min()}")
    print(f"Std absolute error: {res_abs.std()}")
    tree.plot_tree(reg)
    import matplotlib.pyplot as plt
    plt.savefig("tree.png")
    new_all_feat = np.zeros((all_feat.shape[0], all_feat.shape[1] // 4))
    for i in range(all_feat.shape[1] // 4):
        new_all_feat[:, i] = all_feat[:, 4 * i] - all_feat[:, 4 * i + 1] + all_feat[:, 4 * i + 2] - all_feat[:, 4 * i + 3]
    x = lstsq(
        new_all_feat,
        all_values,
    )[0]
    res = all_values - np.dot(new_all_feat, x)
    res_abs = np.abs(res)
    print("Marginal:")
    print(x.T)
    print(f"Mean absolute error: {res_abs.mean()}")
    print(f"Max absolute error: {res_abs.max()}")
    print(f"Min absolute error: {res_abs.min()}")
    print(f"Std absolute error: {res_abs.std()}")
