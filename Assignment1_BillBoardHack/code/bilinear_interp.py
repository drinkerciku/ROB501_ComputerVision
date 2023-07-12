import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    four pixels to compute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')
    
    # Compute the coordinates of the 4 nearest neighbours
    
    ## Find the nearest integer values
    nearestInt = np.rint(pt)
    
    ## Calculate the coordinates of nearest neighbours
    UL = [0,0]
    UR = [0,0]
    LL = [0,0]
    LR = [0,0]
    
    if nearestInt[0][0] < pt[0][0]:
        UL[0] = int(nearestInt[0][0])
        UR[0] = int(nearestInt[0][0] + 1)
        LL[0] = int(nearestInt[0][0])
        LR[0] = int(nearestInt[0][0] + 1)
    else: 
        UL[0] = int(nearestInt[0][0] - 1)
        UR[0] = int(nearestInt[0][0])
        LL[0] = int(nearestInt[0][0] - 1)
        LR[0] = int(nearestInt[0][0])
        
    if nearestInt[1][0] < pt[1][0]:
        UL[1] = int(nearestInt[1][0])
        UR[1] = int(nearestInt[1][0])
        LL[1] = int(nearestInt[1][0] + 1)
        LR[1] = int(nearestInt[1][0] + 1)
    else: 
        UL[1] = int(nearestInt[1][0] - 1)
        UR[1] = int(nearestInt[1][0] - 1)
        LL[1] = int(nearestInt[1][0])
        LR[1] = int(nearestInt[1][0])
    
    # Compute Bilinear Interpolation - algorithm taken from Wikipedia
    V_1 = (UR[0] - pt[0][0])*I[UL[1]][UL[0]] + (pt[0][0] - UL[0])*I[UR[1]][UR[0]]
    V_2 = (LR[0] - pt[0][0])*I[LL[1]][LL[0]] + (pt[0][0] - LL[0])*I[LR[1]][LR[0]]
    
    result = round((LL[1] - pt[1][0])*V_1 + (pt[1][0] - UL[1])*V_2)
    
    # Check if value overflowed due to numerical error
    if result > 255:
        result = 255
    #------------------

    return result