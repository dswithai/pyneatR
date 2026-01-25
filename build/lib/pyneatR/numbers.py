import numpy as np
import warnings
from .utils import check_singleton, sandwich, unique_optimization

def _nround(x, digits=1):
    """
    Round number and return string with fixed decimals.
    """
    return f"{x:.{digits}f}"

def _chunk_digits(x, thousand_separator=","):
    """
    Add separator to number string. 
    x is expected to be a number.
    Returns string.
    """
    pass 

def _format_single_number(n, digits, unit_label, thousand_separator):
    s = f"{n:.{digits}f}"
    
    if thousand_separator:
        parts = s.split('.')
        integer_part = parts[0]
        integer_part = f"{int(integer_part):,}"
        
        if thousand_separator != ',':
            integer_part = integer_part.replace(',', thousand_separator)
            
        s = integer_part
        if len(parts) > 1:
             decimal_mark = "," if thousand_separator == "." else "."
             s += decimal_mark + parts[1]
             
    if s == "0" or s == "0.0" or (s.replace('.','').replace(',','').replace('0','') == ''):
        pass
        
    return s

@unique_optimization
def nnumber(number, digits=1, unit='custom', 
            unit_labels={'thousand': 'K', 'million': 'Mn', 'billion': 'Bn', 'trillion': 'Tn'},
            prefix='', suffix='', thousand_separator=','):
    """
    Neat representation of numbers.
    """
    check_singleton(digits, 'digits', int)
    check_singleton(unit, 'unit', str)
    check_singleton(prefix, 'prefix', str)
    check_singleton(suffix, 'suffix', str)
    
    labels = ['', 
              unit_labels.get('thousand', 'K'),
              unit_labels.get('million', 'Mn'),
              unit_labels.get('billion', 'Bn'),
              unit_labels.get('trillion', 'Tn')]
    factors = [1, 1e-3, 1e-6, 1e-9, 1e-12]
    
    result = []
    
    final_unit_idx = 0
    fixed_unit = False
    
    x_arr = np.asanyarray(number)
    original_shape = x_arr.shape
    x_flat = x_arr.ravel()
    
    if unit == 'auto':
        with np.errstate(divide='ignore'):
            nonzero = x_flat != 0
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
        with np.errstate(divide='ignore'):
             nonzero = uvals != 0
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
        formatted_uvals = sandwich(formatted_uvals, prefix, suffix)
        
    return formatted_uvals[inverse].reshape(original_shape)


@unique_optimization
def npercent(percent, is_decimal=True, digits=1, plus_sign=True, factor_out=False, basis_points_out=False):
    """
    percent: input numbers
    """
    check_singleton(is_decimal, 'is_decimal', bool)
    check_singleton(plus_sign, 'plus_sign', bool)
    
    x = np.asanyarray(percent, dtype=float)
    if is_decimal:
        x = x * 100
        
    fmt_str = f"{{:.{digits}f}}"
    s_list = [fmt_str.format(v) for v in x]
    s_arr = np.array(s_list, dtype=object)
    
    if plus_sign:
        mask_pos = x > 0
        s_arr[mask_pos] = np.char.add("+", s_arr[mask_pos])
        
    s_arr = np.char.add(s_arr, "%")
    
    if factor_out or basis_points_out:
        extras_arr = np.array([""] * len(x), dtype=object)
        
        if factor_out:
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
            
        if basis_points_out:
            bps = x * 100.0
            
            bps_list = [f"{b:+.0f}" for b in bps]
            bps_arr = np.array(bps_list)
            bps_lbl = np.char.add(" (", np.char.add(bps_arr, " bps)"))
            
            extras_arr = np.char.add(extras_arr, bps_lbl)
            
        s_arr = np.char.add(s_arr, extras_arr)
        
    return s_arr
