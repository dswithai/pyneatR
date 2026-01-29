
import numpy as np
from pyneatR import nday, npercent, nnumber, f
from datetime import date

def verify_dtypes():
    print("Verifying dtypes to ensure Windows compatibility...")
    
    # 1. nday
    d = np.array([date.today()])
    res_day = nday(d, show_relative_day=True)
    print(f"nday output dtype: {res_day.dtype}")
    if res_day.dtype.kind not in ('U', 'S'):
         print("❌ nday is returning object or other type!")
    else:
         print("✅ nday is returning native string array.")

    # 2. npercent
    p = np.array([0.5, 0.2])
    res_pct = npercent(p)
    print(f"npercent output dtype: {res_pct.dtype}")
    if res_pct.dtype.kind not in ('U', 'S'):
         print("❌ npercent is returning object or other type!")
    else:
         print("✅ npercent is returning native string array.")

    # 3. f (which uses npercent logic)
    res_f = f(0.5, format_type='percent')
    # f returns scalar for scalar input, let's try array
    res_f_arr = f(np.array([0.5]), format_type='percent')
    print(f"f(percent) output dtype: {res_f_arr.dtype}")
    if res_f_arr.dtype.kind not in ('U', 'S'):
         print("❌ f(percent) is returning object or other type!")
    else:
         print("✅ f(percent) is returning native string array.")

if __name__ == "__main__":
    verify_dtypes()
