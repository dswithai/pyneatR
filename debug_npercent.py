
import numpy as np

def debug_npercent(percent):
    print("--- Debugging npercent ---")
    x = np.asanyarray(percent, dtype=float)
    x = x * 100
    
    fmt_str = "{:.1f}"
    s_list = [fmt_str.format(v) for v in x]
    
    s_arr = np.array(s_list)
    print(f"1. s_arr dtype after init: {s_arr.dtype}")
    
    s_arr = np.where(x > 0, np.char.add("+", s_arr), s_arr)
    print(f"2. s_arr dtype after + sign: {s_arr.dtype}")
    
    s_arr = np.char.add(s_arr, "%")
    print(f"3. s_arr dtype after %: {s_arr.dtype}")
    
    return s_arr

if __name__ == "__main__":
    res = debug_npercent([0.5, 0.2])
    print(f"Final Result dtype: {res.dtype}")
