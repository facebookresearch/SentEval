import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

from argparse import ArgumentParser


def get_scatterplot(
        fi1: np.ndarray, # fi stands for feature imporatnces
        fi2: np.ndarray,
        name1: str="",
        name2: str="",
        task: str="",
        logy: bool=False,
        save_path: str="./output.png",
) -> None:
    outlier_dimensions = [61, 77, 82, 97, 217, 219, 240, 330, 361, 453, 494, 496, 498, 551, 570, 588, 656, 731, 749]
    c = np.zeros((768, 3))
    c[:, 2] = 1
    c[outlier_dimensions] = [1, 0, 0]
    plt.scatter(fi1, fi2, alpha=0.3, c=c, label=rf"$r_s$={sps.spearmanr(fi1, fi2).statistic:.2f}")

    if logy:
        plt.yscale("log")

    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.title(task)
    plt.legend()

    plt.savefig(save_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""
        Draw scatterplot: feature importances of method1 vs feature importances of method2.
        Note: under given fi_path should lie a np.ndarray (i -> feature imortance of i-th feature).
        """
    )
    parser.add_argument("--fi1_path", type=str, help="path to feature importances of method1")
    parser.add_argument("--fi2_path", type=str, help="path to feature importances of method2")
    parser.add_argument("--fi1_name", type=str, default="method1", help="name of method1")
    parser.add_argument("--fi2_name", type=str, default="method2", help="name of method2")
    parser.add_argument("--output_path", type=str, default="./output.png")
    parser.add_argument("--task", type=str, default="", help="probing task")
    parser.add_argument("--logy", action="store_true", help="make yscale logarithmic")
    args = parser.parse_args()

    with open(args.fi1_path, "rb") as file:
        fi1 = np.load(file)

    with open(args.fi2_path, "rb") as file:
        fi2 = np.load(file)

    get_scatterplot(
        fi1, fi2,
        name1=args.fi1_name,
        name2=args.fi2_name,
        task=args.task,
        logy=args.logy,
        save_path=args.output_path,
    )
