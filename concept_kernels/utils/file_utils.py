import json
import logging
import pickle
import sys

import pandas as pd


logger = logging.getLogger(__name__)

def load_csv_as_df(file_path, dtype=None):
    logger.debug("Loading file {} as dataframe".format(file_path))
    if file_path.endswith(".csv"):
        df = pd.read_csv(
            file_path, on_bad_lines="skip", low_memory=True, dtype=dtype
        )
    elif file_path.endswith(".gz"):
        df = pd.read_csv(
            file_path,
            compression="gzip",
            on_bad_lines="skip",
            low_memory=True,
            dtype=dtype,
        )
    return df


def save_df(df, file_path):
    logger.debug("Saving dataframe as CSV file {}".format(file_path))
    df.to_csv(file_path, index=False)


def load_json(file_path):
    logger.debug("Loading JSON file {} as dictionary".format(file_path))
    with open(file_path, "r") as f:
        ret_dict = json.load(f)
    return ret_dict


def save_json(d, file_path):
    logger.debug("Saving dictionary as JSON file {}".format(file_path))
    with open(file_path, "w") as f:
        json.dump(d, f, indent=4)


def load_pickle(file_path):
    logger.debug("Loading data as pickle file {}".format(file_path))
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(d, file_path):
    logger.debug("Saving {} as pickle file {}".format(
        d.__class__.__qualname__, file_path
    ))
    with open(file_path, "wb") as f:
        pickle.dump(d, f)
