from operator import index
import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')
    
    # Create Histogram and Compute the Cumulative Distribution
    histogramEq = np.zeros((1,256), dtype = np.int)
    imgCDF = np.zeros((1,256), dtype = np.float)
    
    ## Retrieve size of the image (height, width)
    dimI = I.shape
    ## Helper Variables
    sizeI = dimI[0] * dimI[1]
    indexI = 0
    sumH = 0
    
    ## Create h(I)
    for i in range(0, dimI[0], 1):
        for j in range(0, dimI[1], 1):
            
            indexI = I[i][j]
            histogramEq[0][indexI] += 1    
            
    ## Compute the cumulative distribution for each intensity value
    for k in range(0, 256, 1):
        
        sumH += histogramEq[0][k]
        imgCDF[0][k] = sumH/sizeI
    
    # Create the contrast-enhanced greyscale intensity image
    
    ## Initialize array of zeros for J
    J = np.zeros(dimI, dtype = np.uint8)
    valueJ = 0.0
    
    ## Populate J
    for i in range(0, dimI[0], 1):
        for j in range(0, dimI[1], 1):
            
            indexI = I[i][j]
            valueJ = np.around(255*imgCDF[0][indexI])
            
            # avoid case of overflowing due to numerical error
            if valueJ > 256:
                valueJ = 255
            
            J[i][j] = valueJ

    #------------------

    return J