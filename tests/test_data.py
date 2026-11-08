import pandas as pd

data = pd.read_csv("diabetes_risk.csv")


def test_columns_are_correct():
    assert list(data.columns) == ["age", "bmi", "glucose", "high_risk"]


def test_no_empty_cells():
    assert data.notna().all().all()


def test_values_are_valid():
    assert (data[["age", "bmi", "glucose"]] > 0).all().all()
    assert set(data["high_risk"].unique()) <= {0, 1}


def test_enough_rows():
    assert len(data) >= 50