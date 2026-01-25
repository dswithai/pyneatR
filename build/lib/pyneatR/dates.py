import numpy as np
import datetime
from .utils import check_singleton, unique_optimization

def _get_weekday_name_vec(dt64):
    """
    Vectorized weekday name extraction.
    """
    days = dt64.astype('datetime64[D]').astype(int)
    idx = (days + 3) % 7
    labels = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    return labels[idx]

@unique_optimization
def nday(date, reference_alias=False):
    """
    Neat alias of the week day.
    """
    check_singleton(reference_alias, 'reference_alias', bool)
    
    if not np.issubdtype(date.dtype, np.datetime64):
         try:
             date = date.astype('datetime64[D]')
         except:
             pass

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
def ndate(date, display_weekday=True, is_month=False):
    """
    Neat representation of dates (vectorized).
    """  
    check_singleton(display_weekday, 'display_weekday', bool)
    check_singleton(is_month, 'is_month', bool)
    
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
def ntimestamp(timestamp, display_weekday=True, include_date=True,
               include_hours=True, include_minutes=True, include_seconds=True,
               include_timezone=True):
    """
    Neat representation of timestamps (Optimized Vectorized).
    """
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
