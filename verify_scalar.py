import numpy as np
from pyneatR import nnumber, nstring

scalar_val = 12345
res_scalar = nnumber(scalar_val)
print(f"Input: {scalar_val}, Type: {type(scalar_val)}")
print(f"Result: {repr(res_scalar)}, Type: {type(res_scalar)}")

if isinstance(res_scalar, str):
    print("PASS: Scalar input returned string.")
else:
    print("FAIL: Scalar input returned array.")

array_val = [12345]
res_array = nnumber(array_val)
print(f"Input: {array_val}, Type: {type(array_val)}")
print(f"Result: {repr(res_array)}, Type: {type(res_array)}")

if isinstance(res_array, np.ndarray):
    print("PASS: List input returned array.")
else:
    print("FAIL: List input returned scalar.")
