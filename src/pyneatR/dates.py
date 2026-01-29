import numpy as np
import datetime
from .utils import check_singleton, unique_optimization
from typing import Union, Any, Optional

def _get_weekday_name_vec(dt64: np.ndarray) -> np.ndarray:
    """
    Vectorized weekday name extraction.

    Parameters
    ----------
    dt64 : numpy.ndarray
        Array of datetime64 values.

    Returns
    -------
    numpy.ndarray
        Array of weekday abbreviations.
    """
    days = dt64.astype('datetime64[D]').astype(int)
    idx = (days + 3) % 7
    labels = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    return labels[idx]

@unique_optimization
def nday(date: Union[np.ndarray, list, datetime.date], reference_alias: bool = False) -> Union[np.ndarray, str]:
    """
    Format dates as day names, optionally with relative alias (Today, Yesterday, etc).

    Parameters
    ----------
    date : array-like
        Input date(s).
    reference_alias : bool, default False
        If True, adds context like 'Today', 'Yesterday', 'Coming', 'Last'.

    Returns
    -------
    numpy.ndarray or str
        Formatted day names or strings.
    """
    check_singleton(reference_alias, 'reference_alias', bool)
    
    if np.size(date) == 0:
        return np.array([], dtype=object)
    
    if not np.issubdtype(getattr(date, 'dtype', None), np.datetime64):
         try:
             date = np.asanyarray(date).astype('datetime64[D]')
         except:
             pass

    if not isinstance(date, np.ndarray):
         date = np.asanyarray(date)

    mask = ~np.isnat(date)
    result = np.full(date.shape, "NaT", dtype=object)
    
    if not np.any(mask):
        return result
        
    valid_dates = date[mask]
    
    day_str = _get_weekday_name_vec(valid_dates)
    
    if reference_alias:
        today = np.datetime64('today')
        d_day = valid_dates.astype('datetime64[D]')
        today_day = today.astype('datetime64[D]')
        
        diff = (today_day - d_day) / np.timedelta64(1, 'D')
        diff = diff.astype(int)
        
        alias = np.full(len(diff), "", dtype=object)
        
        alias[(diff >= 2) & (diff <= 8)] = "Last "
        alias[diff == 1] = "Yesterday, "
        alias[diff == 0] = "Today, "
        alias[diff == -1] = "Tomorrow, "
        alias[(diff >= -8) & (diff <= -2)] = "Coming "
        
        day_str = np.char.add(alias, day_str)
        
    result[mask] = day_str
    return result

@unique_optimization
def ndate(date: Union[np.ndarray, list, datetime.date], display_weekday: bool = True, is_month: bool = False) -> Union[np.ndarray, str]:
    """
    Format dates into a neat string representation.

    Parameters
    ----------
    date : array-like
        Input date(s).
    display_weekday : bool, default True
        If True, appending weekday name e.g. (Mon).
    is_month : bool, default False
        If True, formats as month abbreviation and year (Jan'23).

    Returns
    -------
    numpy.ndarray or str
        Formatted date strings.
    """  
    check_singleton(display_weekday, 'display_weekday', bool)
    check_singleton(is_month, 'is_month', bool)
    
    if np.size(date) == 0:
        return np.array([], dtype=object)
    
    if not np.issubdtype(getattr(date, 'dtype', None), np.datetime64):
         try:
             date = np.asanyarray(date).astype('datetime64[D]')
         except:
             # Fallback attempt if astype fails or if it's already object
             date = np.asanyarray(date)

    if not isinstance(date, np.ndarray):
        date = np.asanyarray(date)
        
    mask = ~np.isnat(date)
    result = np.full(date.shape, "NaT", dtype=object)
    
    if not np.any(mask):
        return result
        
    valid_dates = date[mask]
    
    iso = np.datetime_as_string(valid_dates, unit='D')
    
    if is_month:
        months_full = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months_arr = np.array(months_full)
        
        mm = np.array([s[5:7] for s in iso], dtype=int)
        yy = np.array([s[2:4] for s in iso])
        
        mon_str = months_arr[mm]
        
        s = np.char.add(np.char.add(mon_str, "'"), yy)
        
    else:
        months_full = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months_arr = np.array(months_full)
        
        yyyy = np.array([s[0:4] for s in iso])
        mm = np.array([s[5:7] for s in iso], dtype=int)
        dd = np.array([s[8:10] for s in iso])
        
        mon_str = months_arr[mm]
        
        p1 = np.char.add(mon_str, " ")
        p2 = np.char.add(p1, dd)
        p3 = np.char.add(p2, ", ")
        s = np.char.add(p3, yyyy)
        
        if display_weekday:
            wd = _get_weekday_name_vec(valid_dates)
            w_part = np.char.add(" (", np.char.add(wd, ")"))
            s = np.char.add(s, w_part)
            
    result[mask] = s
    return result

@unique_optimization
def ntimestamp(timestamp: Union[np.ndarray, list, datetime.datetime], 
               display_weekday: bool = True, include_date: bool = True,
               include_hours: bool = True, include_minutes: bool = True, include_seconds: bool = True,
               include_timezone: bool = True) -> Union[np.ndarray, str]:
    """
    Format timestamps into a neat string representation.

    Parameters
    ----------
    timestamp : array-like
        Input timestamp(s).
    display_weekday : bool, default True
        If True, appending weekday name e.g. (Mon).
    include_date : bool, default True
        If True, include date part.
    include_hours : bool, default True
        If True, include hours (12H format).
    include_minutes : bool, default True
        If True, include minutes.
    include_seconds : bool, default True
        If True, include seconds.
    include_timezone : bool, default True
        Reserved parameter for future timezone support (currently unused).

    Returns
    -------
    numpy.ndarray or str
        Formatted timestamp strings.
    """
    if np.size(timestamp) == 0:
        return np.array([], dtype=object)

    if not isinstance(timestamp, np.ndarray):
         timestamp = np.asanyarray(timestamp)
         if not np.issubdtype(timestamp.dtype, np.datetime64):
             try:
                 timestamp = timestamp.astype('datetime64[s]')
             except:
                 pass

    mask = ~np.isnat(timestamp)
    result = np.full(timestamp.shape, "NaT", dtype=object)
    
    if not np.any(mask):
        return result
        
    valid_ts = timestamp[mask]
    
    iso = np.datetime_as_string(valid_ts, unit='s')
    
    parts_list = []
    
    if include_date:
        months_full = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months_arr = np.array(months_full)
        
        yyyy = np.array([s[0:4] for s in iso])
        mm = np.array([s[5:7] for s in iso], dtype=int)
        dd = np.array([s[8:10] for s in iso])
        mon_str = months_arr[mm]
        
        d_str = np.char.add(mon_str, " ")
        d_str = np.char.add(d_str, dd)
        d_str = np.char.add(d_str, ", ")
        d_str = np.char.add(d_str, yyyy)
        d_str = np.char.add(d_str, " ")
        
        parts_list.append(d_str)

    hh_str = np.array([s[11:13] for s in iso])
    mm_str = np.array([s[14:16] for s in iso])
    ss_str = np.array([s[17:19] for s in iso])
    
    hh_int = hh_str.astype(int)
    is_pm = hh_int >= 12
    
    hh_12 = hh_int.copy()
    hh_12[hh_12 == 0] = 12
    hh_12[hh_12 > 12] -= 12
    
    hours_labels = np.array([f"{i:02d}" for i in range(13)])
    hh_12_str = hours_labels[hh_12]
    
    if include_hours:
        parts_list.append(np.char.add(hh_12_str, "H"))
        
    if include_minutes:
        parts_list.append(np.char.add(" ", np.char.add(mm_str, "M")))
        
    if include_seconds:
        parts_list.append(np.char.add(" ", np.char.add(ss_str, "S")))
        
    if include_timezone:
        pass
        
    ampm_arr = np.where(is_pm, " PM", " AM")
    parts_list.append(ampm_arr)
    
    combined = parts_list[0]
    for p in parts_list[1:]:
        combined = np.char.add(combined, p)
        
    if display_weekday:
        wd = _get_weekday_name_vec(valid_ts)
        w_part = np.char.add(" (", np.char.add(wd, ")"))
        combined = np.char.add(combined, w_part)
        
    result[mask] = combined
    return result
