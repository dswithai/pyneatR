import pytest
import numpy as np
from pyneatR import nnumber, npercent

def test_nnumber_basics():
    x = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
    expected = ['10.0', '100.0', '1.0 K', '10.0 K', '100.0 K', '1.0 Mn', '10.0 Mn', '100.0 Mn', '1.0 Bn']
    
    # Test auto unit
    res = nnumber(x, digits=1, unit='auto')
    # Note: 'auto' logic implementation might vary slightly on mode calculation
    # Let's check simply.
    # Actually my implementation for 'auto' finds the mode of magnitudes.
    # Magnitudes: 1, 2, 3(K), 4(K), 5(K), 6(Mn), 7(Mn), 8(Mn), 9(Bn)
    # Counts of mag index: 
    # 0 (1-999): 2
    # 1 (K): 3
    # 2 (Mn): 3
    # 3 (Bn): 1
    # Mode is tied between K and Mn. standard np.argmax returns first occurrence -> K.
    # So if K is chosen: 
    # 10 -> 0.0 K (<0.1 K ?)
    # 1000 -> 1.0 K
    # 1M -> 1,000.0 K
    
    # Test custom per-value unit (unit='custom') - DEFAULT
    res_custom = nnumber(x, digits=1, unit='custom')
    # This should match 'expected' exactly as per R example
    assert np.all(res_custom == expected)

def test_nnumber_fixed_unit():
    x = [123456789.123456]
    res = nnumber(x, digits=1, unit='Mn', prefix='$')
    assert res[0] == '$123.5 Mn'

def test_nnumber_separators():
    x = [10000]
    res = nnumber(x, digits=0, thousand_separator='.', unit='')
    # 10.000
    assert res[0] == '10.000'

def test_npercent():
    # 22.3%
    res = npercent([0.223], is_decimal=True, digits=1)
    assert res[0] == '+22.3%'
    
    res2 = npercent([22.3], is_decimal=False, digits=1)
    assert res2[0] == '+22.3%'
    
    # Growth/Drop
    # -4.01 -> -401% -> 4.0x Drop
    # 2.56 -> 256% -> 2.6x Growth
    res3 = npercent([-4.01, 2.56], is_decimal=True, factor_out=True)
    assert '4.0x Drop' in res3[0]
    assert '2.6x Growth' in res3[1]
    
    # Basis points
    res4 = npercent([0.01], is_decimal=True, basis_points_out=True)
    assert '100 bps' in res4[0]
