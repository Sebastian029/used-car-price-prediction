import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_prep import load_clean, parse_number, parse_torque  # noqa: E402


@pytest.mark.parametrize("raw, expected", [
    ("23.4 kmpl", 23.4),
    ("1248 CC", 1248),
    ("73.9 bhp", 73.9),
    ("26.6 km/kg", 26.6),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [" bhp", "0", None, np.nan])
def test_parse_number_missing(raw):
    assert np.isnan(parse_number(raw))


@pytest.mark.parametrize("raw, expected", [
    ("190Nm@ 2000rpm", 190.0),
    ("200Nm@ 1400-3400rpm", 200.0),
    ("22.4 kgm at 1750-2750rpm", 219.7),     # kgm converted to Nm
    ("12.7@ 2,700(kgm@ rpm)", 124.5),
    ("115@ 2,500(kgm@ rpm)", 115.0),         # mislabelled as kgm, really Nm
    ("380Nm(38.7kgm)@ 2500rpm", 380.0),      # both units given, Nm wins
    ("250@ 1250-5000rpm", 250.0),            # no unit, taken as Nm
])
def test_parse_torque(raw, expected):
    assert parse_torque(raw) == pytest.approx(expected)


def test_load_clean():
    df = load_clean()
    # Exact duplicates (~15% of raw rows) are removed. A handful of rows stay
    # identical after 'name' is reduced to brand + model: different trims of the same car.
    assert df.duplicated().mean() < 0.01
    assert df["torque"].max() < 1000
    assert df["selling_price"].gt(0).all()
    assert {"brand", "model"} <= set(df.columns)
