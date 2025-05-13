import numpy as np

def fwhm(x: np.ndarray, y: np.ndarray, height: float = 0.5) -> float:
    """
    Gives the full-width at half-maximum of data described by x and y arrays.

    Args:
        x (np.ndarray): The x-values (e.g. the scale of the vector; has the same units as the return value)
        y (np.ndarray): The y-values (to which half-maximum refers)
        height (float): Instead of half-max, can optionally return height*max. Default is 0.5.

    Returns:
        float: The full-width at half-max (units of x)
    """
    heightLevel = np.max(y) * height
    indexMax = np.argmax(y)
    y = np.roll(y, - indexMax + int(np.shape(y)[0]/2),axis=0)
    indexMax = np.argmax(y)
    xLower = np.interp(heightLevel, y[:indexMax], x[:indexMax])
    xUpper = np.interp(heightLevel, np.flip(y[indexMax:]), np.flip(x[indexMax:]))
    return xUpper - xLower
