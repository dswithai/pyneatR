import pytest
import numpy as np
from pyneatR import nstring

def test_nstring_case():
    x = ["hello world", "GOODBYE"]
    res = nstring(x, case='title')
    assert res[0] == "Hello World"
    assert res[1] == "Goodbye"
    
def test_nstring_specials():
    x = ["hello!!! world@"]
    res = nstring(x, remove_specials=True)
    # keep only alnum + space
    assert res[0] == "hello world"

def test_nstring_start_case():
    x = ["all Models are Wrong"]
    res = nstring(x, case='start')
    # All Models Are Wrong
    assert res[0] == "All Models Are Wrong"
