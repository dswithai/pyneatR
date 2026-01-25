import pytest
import numpy as np
import pandas as pd
import polars as pl
from pyneatR import nnumber, ndate, nstring

def test_pandas_series():
    # Numbers
    s = pd.Series([1000, 2000, 3000])
    res = nnumber(s)
    # Should return numpy array of strings
    assert isinstance(res, np.ndarray) or isinstance(res, list) or isinstance(res, pd.Series)
    assert res[0] == "1.0 K"
    
    # Dates
    d = pd.to_datetime(["2023-01-01", "2023-01-02"])
    res_d = ndate(d, display_weekday=False)
    assert res_d[0] == "Jan 01, 2023"

def test_pandas_dataframe_column():
    df = pd.DataFrame({"vals": [1000000, 2000000]})
    res = nnumber(df["vals"], unit='Mn')
    assert res[0] == "1.0 Mn"

def test_polars_series():
    s = pl.Series("vals", [1000, 2000, 3000])
    # nnumber handles numpy-like. Polars series might need conversion or nnumber handles it via np.asanyarray
    res = nnumber(s)
    
    # If nnumber returns numpy array, that's fine.
    # We check content.
    assert "1.0 K" in res[0] 

def test_polars_edge_cases():
    # Polars with nulls
    s = pl.Series("vals", [1000, None, 3000])
    res = nnumber(s)
    assert len(res) == 3
