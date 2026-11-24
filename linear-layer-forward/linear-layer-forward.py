def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Write code here
    f1 = np.dot(X,W) + b 
    return f1.tolist()
    