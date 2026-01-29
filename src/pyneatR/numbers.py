import numpy as np
import warnings
from .utils import _check_singleton, _sandwich, _unique_optimization
from typing import Union, Any, Optional, Dict

def _nround(x: float, digits: int = 1) -> str:
    """
    Round number and return string with fixed decimals.
    """
    return f"{x:.{digits}f}"



@_unique_optimization
def nnumber(number: Union[np.ndarray, list, float, int], digits: int = 1, unit: str = 'custom', 
            unit_labels: Dict[str, str] = {'thousand': 'K', 'million': 'Mn', 'billion': 'Bn', 'trillion': 'Tn'},
            prefix: str = '', suffix: str = '', thousand_separator: str = ',') -> Union[np.ndarray, str]:
    """
    Neat representation of numbers with optional units (K, Mn, Bn) and formatting.

    Parameters
    ----------
    number : array-like
        Input numbers.
    digits : int, default 1
        Number of decimal digits to display.
    unit : str, default 'custom'
         'auto': Automatically determine best unit for all numbers.
         'custom': Determine best unit for each number individually.
         'K', 'Mn', 'Bn', 'Tn': Fix unit to Thousand, Million, Billion, Trillion.
         '': No unit.
    unit_labels : dict, optional
        Labels for units (thousand, million, billion, trillion).
    prefix : str, default ''
        Prefix string (e.g. '$').
    suffix : str, default ''
        Suffix string (e.g. ' USD').
    thousand_separator : str, default ','
        Character for thousand separation.

    Returns
    -------
    numpy.ndarray or str
        Formatted number strings.
    """
    _check_singleton(digits, 'digits', int)
    _check_singleton(unit, 'unit', str)
    _check_singleton(prefix, 'prefix', str)
    _check_singleton(suffix, 'suffix', str)
    
    labels = ['', 
              unit_labels.get('thousand', 'K'),
              unit_labels.get('million', 'Mn'),
              unit_labels.get('billion', 'Bn'),
              unit_labels.get('trillion', 'Tn')]
    factors = [1, 1e-3, 1e-6, 1e-9, 1e-12]
    
    result = []
    
    final_unit_idx = 0
    fixed_unit = False
    
    x_arr = np.asanyarray(number, dtype=float)
    original_shape = x_arr.shape
    x_flat = x_arr.ravel()
    
    if unit == 'auto':
        with np.errstate(divide='ignore', invalid='ignore'):
            nonzero = (x_flat != 0) & np.isfinite(x_flat)
            if not np.any(nonzero):
                mode_mx = 0
            else:
                logs = np.log10(np.abs(x_flat[nonzero]))
                mx = np.floor(logs / 3).astype(int)
                limit = len(labels) - 1
                mx = np.clip(mx, 0, limit)
                counts = np.bincount(mx)
                mode_mx = np.argmax(counts)
        
        final_unit_idx = mode_mx
        fixed_unit = True
        
    elif unit == 'custom':
        fixed_unit = False
        
    elif unit in labels:
        final_unit_idx = labels.index(unit)
        fixed_unit = True
    else:
         raise ValueError("Invalid unit")

    uvals, inverse = np.unique(x_flat, return_inverse=True)
    
    if fixed_unit:
        indices = np.full(len(uvals), final_unit_idx, dtype=int)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
             nonzero = (uvals != 0) & np.isfinite(uvals)
             indices = np.zeros(len(uvals), dtype=int)
             if np.any(nonzero):
                 logs = np.log10(np.abs(uvals[nonzero]))
                 vals = np.floor(logs / 3).astype(int)
                 limit = len(labels) - 1
                 indices[nonzero] = np.clip(vals, 0, limit)
    
    factors_arr = np.array(factors)
    labels_arr = np.array(labels)
    
    scales = factors_arr[indices]
    unit_lbls = labels_arr[indices]
    
    scaled_vals = uvals * scales
    
    fmt_str = f"{{:,.{digits}f}}"
    formatted_list = [fmt_str.format(v) for v in scaled_vals]
    formatted_arr = np.array(formatted_list)
    
    if thousand_separator != ',':
        if thousand_separator == '.':
            formatted_arr = np.char.replace(formatted_arr, ',', 'X')
            formatted_arr = np.char.replace(formatted_arr, '.', ',')
            formatted_arr = np.char.replace(formatted_arr, 'X', '.')
        else:
            formatted_arr = np.char.replace(formatted_arr, ',', thousand_separator)

    has_unit = unit_lbls != ""
    if np.any(has_unit):
        suffixes = np.where(has_unit, np.char.add(" ", unit_lbls), "")
        formatted_arr = np.char.add(formatted_arr, suffixes)
    
    formatted_uvals = formatted_arr

    if prefix or suffix:
        formatted_uvals = _sandwich(formatted_uvals, prefix, suffix)
        
    return formatted_uvals[inverse].reshape(original_shape)


@_unique_optimization
def npercent(percent: Union[np.ndarray, list, float, int], is_ratio: bool = True, digits: int = 1, 
             show_plus_sign: bool = True, show_growth_factor: bool = False, show_bps: bool = False) -> Union[np.ndarray, str]:
    """
    Format numbers as percentages.

    Parameters
    ----------
    percent : array-like
        Input numbers.
    is_ratio : bool, default True
        If True, input is treated as ratio (0.1 -> 10%).
        If False, input is treated as whole percentage (10 -> 10%).
    digits : int, default 1
        Number of decimal digits to display.
    show_plus_sign : bool, default True
        If True, prepend '+' to positive values.
    show_growth_factor : bool, default False
        If True, add growth factor (e.g. 2x Growth).
    show_bps : bool, default False
        If True, add basis points label (e.g. 100 bps).

    Returns
    -------
    numpy.ndarray or str
        Formatted percentage strings.
    """
    _check_singleton(is_ratio, 'is_ratio', bool)
    _check_singleton(show_plus_sign, 'show_plus_sign', bool)
    
    x = np.asanyarray(percent, dtype=float)
    if is_ratio:
        x = x * 100
        
    if len(x) == 0:
        return np.array([], dtype=object)
        
    fmt_str = f"{{:.{digits}f}}"
    s_list = [fmt_str.format(v) for v in x]
    fmt_str = f"{{:.{digits}f}}"
    s_list = [fmt_str.format(v) for v in x]
    s_arr = np.array(s_list)
    
    if show_plus_sign:
        s_arr = np.where(x > 0, np.char.add("+", s_arr), s_arr)
        
    s_arr = np.char.add(s_arr, "%")
    
    if show_growth_factor or show_bps:
        extras_arr = np.array([""] * len(x))
        
        if show_growth_factor:
            gtemp = x / 100.0
            gtemp_abs = np.abs(gtemp)
            
            g_fmt = [f"{v:.1f}" for v in gtemp_abs]
            g_fmt_arr = np.array(g_fmt)
            
            growth_lbl = np.char.add(" (", np.char.add(g_fmt_arr, "x Growth)"))
            drop_lbl = np.char.add(" (", np.char.add(g_fmt_arr, "x Drop)"))
            
            small_lbl = np.where(gtemp > 0, " (Growth)", " (Drop)")
            small_lbl = np.where(gtemp == 0, " (Flat)", small_lbl)
            
            f_lbl = np.where(gtemp <= -1, drop_lbl, small_lbl)
            f_lbl = np.where(gtemp >= 1, growth_lbl, f_lbl)
            
            extras_arr = np.char.add(extras_arr, f_lbl)
            
        if show_bps:
            bps = x * 100.0
            
            bps_list = [f"{b:+.0f}" if b != 0 else "0" for b in bps]
            bps_arr = np.array(bps_list)
            bps_lbl = np.char.add(" (", np.char.add(bps_arr, " bps)"))
            
            extras_arr = np.char.add(extras_arr, bps_lbl)
            
            extras_arr = np.char.add(extras_arr, bps_lbl)
            
        s_arr = np.char.add(s_arr, extras_arr)
        
    return s_arr
