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
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn.model_selection
import sklearn.linear_model
import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)

def oneHotFeatures(X):
        hour = X[:, 3].astype(int)
        hour_one_hot = np.eye(24)[hour]

        month = X[:, 2].astype(int)
        month_one_hot = np.eye(12)[month - 1]

        workingday = X[:, 6:7]
        hour_workingday = hour_one_hot * workingday

        day = X[:, 5].astype(int)
        day_one_hot = np.eye(7)[day]

        weather = X[:, 7].astype(int)
        weather_one_hot = np.eye(4)[weather - 1]

        season = X[:, 0].astype(int)
        season_one_hot = np.eye(4)[season - 1]

        temp = X[:, 8:9]
        weather_temp_interaction = weather_one_hot * temp

        X_new = np.column_stack((season_one_hot, X[:, 1:2], month_one_hot, hour_one_hot, hour_workingday,  X[:, 4:5], day_one_hot, X[:, 6:7], weather_one_hot, weather_temp_interaction, X[:,8:]))
        return X_new

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        data_train = train.data
        target_train = train.target

        model = sklearn.linear_model.LinearRegression()
        model.fit(data_train, target_train)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        data_test = test.data

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        predictions = model.predict(data_test)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
