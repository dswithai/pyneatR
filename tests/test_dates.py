import pytest
import numpy as np
import datetime
from pyneatR import ndate, ntimestamp, nday

def test_ndate():
    d = np.array(['2023-01-01', '2023-01-02'], dtype='datetime64[D]')
    res = ndate(d, display_weekday=False)
    assert res[0] == "Jan 01, 2023"

def test_ndate_weekday():
    d = np.array(['2023-01-01'], dtype='datetime64[D]') # Sunday
    res = ndate(d, display_weekday=True)
    assert "Sun" in res[0]

def test_ntimestamp():
    # 2023-01-01 12:30:45
    ts = np.array(['2023-01-01T12:30:45'], dtype='datetime64[s]')
    res = ntimestamp(ts, include_timezone=False)
    # Expected: Jan 01, 2023 12H 30M 45S PM (Sun)
    # Note: 12:30:45 -> 12H 30M 45S PM
    result = res[0]
    assert "Jan 01, 2023" in result
    assert "12H" in result
    assert "30M" in result
    assert "45S" in result
    assert "PM" in result

def test_nday_alias():
    # Helper to control 'today' reference is hard because logic uses np.datetime64('today')
    # So we test non-relative first.
    d = np.array(['2023-01-01'], dtype='datetime64[D]')
    res = nday(d, reference_alias=False)
    assert res[0] == "Sun"
    
    # Relative alias test - risky if 'today' changes, but relative logic:
    # If we pass today
    today = np.datetime64('today')
    res_today = nday([today], reference_alias=True)
    assert "Today" in res_today[0]
    
    yesterday = today - np.timedelta64(1, 'D')
    res_yest = nday([yesterday], reference_alias=True)
    assert "Yesterday" in res_yest[0]
