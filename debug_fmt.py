digits = 1
v = 22.3
fmt_str = f"{{:.{digits}f}}"
print(f"fmt_str: {fmt_str}")
res = fmt_str.format(v)
print(f"res: {res}")

# Alternative
res2 = f"{v:.{digits}f}"
print(f"res2: {res2}")
