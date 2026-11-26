import math
import numpy as np

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    L = math.sqrt(6 / (fan_in + fan_out))

    def scale(X):
        if isinstance(X, list):
            return [scale(v) for v in X]   # recurse into rows
        return X * 2 * L - L               # number or numpy array

    return scale(W)