import numpy as np
import re
from .utils import _check_singleton, _unique_optimization
from typing import Union, Optional

def _clean_text_single(text: str, whitelist_specials: str = "") -> str:
    if not text:
        return ""
    escaped_whitelist = re.escape(whitelist_specials)
    pattern = f"[^a-zA-Z0-9\\s{escaped_whitelist}]"
    return re.sub(pattern, "", text)

def _clean_space_single(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def _strip_non_english_single(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^\x20-\x7E]", "", text)

def _convert_case_single(text: str, case: str) -> str:
    if case == 'lower':
        return text.lower()
    elif case == 'upper':
        return text.upper()
    elif case == 'title':
        return text.title()
    elif case == 'start':
        return _string_start_case(text)
    elif case == 'initcap':
        return text.capitalize()
    return text

def _string_start_case(text: str) -> str:
    return ' '.join(word.capitalize() for word in text.split())

@_unique_optimization
def nstring(x: Union[np.ndarray, list, str], case: Optional[str] = None, 
            remove_specials: bool = False, whitelist_specials: str = '', en_only: bool = False) -> Union[np.ndarray, str]:
    """
    Neat representation of strings with case conversion and cleaning.
    
    Parameters
    ----------
    x : array-like
        Input strings.
    case : {'lower', 'upper', 'title', 'start', 'initcap'}, optional
        Case conversion:
        - 'lower': all lowercase
        - 'upper': all uppercase
        - 'title': Title Case
        - 'start': Start Case (First letter of each word capitalized)
        - 'initcap': Initcap (First letter of string capitalized)
    remove_specials : bool, default False
        If True, remove special characters (non-alphanumeric/whitespace).
    whitelist_specials : str, optional
        Characters to keep when remove_specials is True.
    en_only : bool, default False
        If True, keep only English (ASCII) characters.
    
    Returns
    -------
    numpy.ndarray or str
        Formatted strings.
    """
    _check_singleton(remove_specials, 'remove_specials', bool)
    _check_singleton(en_only, 'en_only', bool)
    
    result = []
    
    clean_text_pattern = None
    if remove_specials:
        escaped_whitelist = re.escape(whitelist_specials)
        clean_text_pattern = re.compile(f"[^a-zA-Z0-9\\s{escaped_whitelist}]")
        
    non_english_pattern = None
    if en_only:
        non_english_pattern = re.compile(r"[^\x20-\x7E]")
        
    space_pattern = re.compile(r"\s+")

    for s in x:
        s = str(s)
        
        if case:
            s = _convert_case_single(s, case)
            
        if remove_specials:
            if clean_text_pattern:
                s = clean_text_pattern.sub("", s)
        
        if en_only:
            if non_english_pattern:
                s = non_english_pattern.sub("", s)
                
        s = space_pattern.sub(" ", s).strip()
        
        result.append(s)
        
    return np.array(result, dtype=object)
