import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def get_scatterplot(
        fi1: np.ndarray, # fi stands for feature imporatnces
        fi2: np.ndarray,
        name1: str="",
        name2: str="",
        save_path: str="./output.png",
) -> None:
    plt.scatter(fi1, fi2)
    plt.xlabel(name1)
    plt.ylabel(name2)
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
    args = parser.parse_args()

    with open(args.fi1_path, "rb") as file:
        fi1 = np.load(file)

    with open(args.fi2_path, "rb") as file:
        fi2 = np.load(file)

    get_scatterplot(
        fi1, fi2,
        name1=args.fi1_name,
        name2=args.fi2_name,
        save_path=args.output_path,
    )