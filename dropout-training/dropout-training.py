import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    if rng is None:
          rng = np.random.default_rng()
    mask = rng.random(np.asarray(x).shape) >= p
    dropout_pattern =  mask * (0 if p == 1 else 1 / (1 - p))
    return x*dropout_pattern,dropout_pattern
    