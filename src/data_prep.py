"""Load the raw CarDekho 'Car details v3' CSV and turn it into model-ready columns.

Only row-wise parsing happens here. Anything that learns from the data
(imputation, scaling, encoding) lives in the sklearn pipeline in models.py,
so it is fitted on the training split only.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "car_details_v3.csv"

KGM_TO_NM = 9.80665
MAX_NM_PER_CC = 0.4
TARGET = "selling_price"

NUMERIC_FEATURES = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats"]
CATEGORICAL_FEATURES = ["brand", "model", "fuel", "seller_type", "transmission", "owner"]

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def parse_number(value):
    """First number in a string such as '23.4 kmpl', '1248 CC' or '73.9 bhp'."""
    if not isinstance(value, str):
        return np.nan
    match = _NUMBER.search(value.replace(",", ""))
    if not match:
        return np.nan
    number = float(match.group())
    return number if number > 0 else np.nan


def parse_torque(value):
    """Torque in Nm from the many formats in the data.

    Examples: '190Nm@ 2000rpm', '22.4 kgm at 1750-2750rpm',
    '12.7@ 2,700(kgm@ rpm)', '380Nm(38.7kgm)@ 2500rpm'.

    The '(kgm@ rpm)' label is sometimes wrong in the source data
    ('115@ 2,500(kgm@ rpm)' is 115 Nm, not 1128 Nm). Car torque in kgm is
    roughly 5-60, so only values below 100 are converted from kgm.
    """
    number = parse_number(value)
    if np.isnan(number):
        return np.nan
    text = value.lower()
    if "nm" not in text and "kgm" in text and number < 100:
        return round(number * KGM_TO_NM, 1)
    return number


def load_clean(path=RAW_PATH):
    """Return the cleaned DataFrame: parsed numeric columns, brand/model, no duplicates."""
    df = pd.read_csv(path)

    # About 15% of rows are exact duplicates. With a random split the same
    # listing would end up in both train and test and inflate the scores.
    df = df.drop_duplicates().reset_index(drop=True)

    for col in ["mileage", "engine", "max_power"]:
        df[col] = df[col].map(parse_number)
    df["torque"] = df["torque"].map(parse_torque)

    # Source typos such as a 1.5 L Maruti Zen listed with 789 Nm. The highest
    # genuine ratio in the data is ~0.33 Nm/cc (Volvo XC90 T8 hybrid).
    implausible = df["torque"] / df["engine"] > MAX_NM_PER_CC
    df.loc[implausible, "torque"] = np.nan

    # 'name' has ~2000 unique values; brand + model generalise much better.
    name_parts = df["name"].str.split()
    df["brand"] = name_parts.str[0]
    df["model"] = name_parts.str[0] + " " + name_parts.str[1].fillna("")

    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]


def train_test(test_size=0.2, random_state=42):
    """The one fixed split used by every script. The test set is only scored in train.py."""
    df = load_clean()
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    data = load_clean()
    print(data.shape)
    print(data.isna().sum())
    print(data.head())
