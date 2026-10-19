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
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=92, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.hstack((data, np.ones(shape=(args.data_size, 1)) ))

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights.
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `len(train_data)`.
        #
        # The gradient for the input example $(x_i, t_i)$ is
        # - $(x_i^T weights - t_i) x_i$ for the unregularized loss (1/2 MSE loss),
        # - $args.l2 * weights_with_bias_set_to_zero$ for the L2 regularization loss,
        #   where we set the bias to zero because the bias should not be regularized,
        # and the SGD update is
        #   weights = weights - args.learning_rate * gradient


        for i in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[i: i + args.batch_size]
            gradient = np.matmul((train_data[batch_indices]), weights) - train_target[batch_indices]
            gradient = np.matmul(train_data[batch_indices].T, gradient)
            weights_with_bias_set_to_zero = weights.copy()
            weights_with_bias_set_to_zero[-1] = 0

            l2 = args.l2 * weights_with_bias_set_to_zero

            gradient = gradient/args.batch_size + l2

            weights -= args.learning_rate * gradient


        # TODO: Append current RMSE on train/test to `train_rmses`/`test_rmses`.

        predicted_train_target = np.matmul(train_data, weights)
        train_mse = sklearn.metrics.mean_squared_error(y_true = train_target, y_pred = predicted_train_target)

        predicted_test_target = np.matmul(test_data, weights)
        test_mse = sklearn.metrics.mean_squared_error(y_true = test_target, y_pred = predicted_test_target)

        train_rmses.append(np.sqrt(train_mse))

        test_rmses.append(np.sqrt(test_mse))

    # TODO: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression()` on `train_data` (ignoring `args.l2`).

    model = sklearn.linear_model.LinearRegression()
    
    regression = model.fit(train_data, train_target)

    prediction = regression.predict(test_data)

    mse = sklearn.metrics.mean_squared_error(y_true = test_target, y_pred = prediction)
    explicit_rmse = np.sqrt(mse)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(main_args)
    print("Test RMSE: SGD {:.3f}, explicit {:.1f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.3f}".format(weight) for weight in weights[:12]), "...")
