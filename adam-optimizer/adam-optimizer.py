import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m_corrected = np.zeros(m.shape)
    v_corrected = np.zeros(v.shape)
    m =  beta1 * m + (1-beta1) * grad
    v =  beta2 * v + (1-beta2) * grad**2
    m_corrected  = m / (1-beta1**t)
    v_corrected  = v  / (1-beta2**t)
    param = param - lr * (m_corrected /(np.sqrt(v_corrected)+eps))
    return param,m,v