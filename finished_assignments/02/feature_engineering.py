#!/usr/bin/env python3
#
# All team solutions **must** list **all** members of the team.
# The members must be listed using their ReCodEx IDs anywhere
# in a comment block in the source file (on a line beginning with `#`).
#
# You can find out ReCodEx ID in the URL bar after navigating
# to your User profile page. The ID has the following format:
# 01234567-89ab-cdef-0123-456789abcdef.
#
# Denis Šmerda
# e198bace-6fb1-43c6-8a4e-403d47252059
# Shtanko Kyrylo
# 5f81d778-9180-44aa-bf06-f6750bd0cead
# Hulianskyi Oleksandr
# 2e4b040f-8cc3-4418-876c-6364f010b60b

import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    categorical_columns = [
        i for i in range(train_data.shape[1])
        if (np.all(train_data[:, i] == np.floor(train_data[:, i])))
    ]

    numerical_columns = [i for i in range(train_data.shape[1]) if i not in categorical_columns]

    encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    scaler = sklearn.preprocessing.StandardScaler()

    preproces = sklearn.compose.ColumnTransformer(transformers=[("categorical", encoder, categorical_columns), ("numerical", scaler, numerical_columns)],  remainder="drop")
    train_data = preproces.fit_transform(train_data)
    test_data = preproces.transform(test_data)

    polynomialFeatures = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

    train_data = polynomialFeatures.fit_transform(train_data)
    test_data = polynomialFeatures.transform(test_data)

    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(main_args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))),
                  *["..."] if dataset.shape[1] > 140 else [])
