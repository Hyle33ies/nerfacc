"""
Utility functions for dataset handling.
"""

from collections import namedtuple
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

# Define a namedtuple for ray representation
Rays = namedtuple("Rays", ("origins", "directions"))


def namedtuple_map(fn: Callable, tup: Tuple) -> Tuple:
    """Apply a function to each element of a namedtuple and return a new namedtuple."""
    return type(tup)(*map(fn, tup)) 
