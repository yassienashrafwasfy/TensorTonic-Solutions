import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    _,counts = np.unique(y,return_counts=True)
    entropy  = counts / len(y)
    return - np.sum(entropy *np.log2(entropy)) 