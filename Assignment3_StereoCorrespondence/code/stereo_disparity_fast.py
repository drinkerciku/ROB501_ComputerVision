import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---

    # Initialize output to zeros    
    Id = np.zeros(Il.shape)
    numNeigh = 4
    # Search Parameters
    startX = bbox[0][0] + numNeigh
    startY = bbox[1][0] + numNeigh
    endX = bbox[0][1] + numNeigh
    endY = bbox[1][1] + numNeigh
    # Helper Variables
    prevSAD = 0
    currSAD = 0
    currDisp = 0
    currWindL = None
    currWindR = None
    
    # apply a gaussian_laplace filter to emphasise contours
    Il = gaussian_laplace(Il, sigma = 0.5)
    Ir = gaussian_laplace(Ir, sigma = 0.5)
    
    # Search only withing the bounding box specified on the left image
    for i in range(startY, endY, 1):
        for j in range(startX, endX , 1):
            
            # get current window from Il
            currWindL = Il[(i - numNeigh) : (i + numNeigh + 1), (j - numNeigh) : (j + numNeigh + 1)]
            
            # loop over the distance dmax on the second image
            # only in one direction on epipolar lines
            currDisp = 0
            prevSAD = 10e14
            for k in range(0, maxd + 1, 1):
                
                # get current window from Ir and compute SAD score if 
                # we are within the bounds of the image
                if (j - k - numNeigh) < 0 or (j + numNeigh - k + 1) >= Il.shape[1]:
                    continue
                
                currWindR = Ir[(i - numNeigh) : (i + numNeigh + 1), (j - numNeigh - k) : (j + numNeigh - k + 1)]
                currSAD = np.sum(np.abs(currWindL - currWindR)) # SAD Metric
                
                # update SAD score
                if currSAD < prevSAD:
                    prevSAD = currSAD
                    currDisp = k
            
            # assign disparity value
            Id[i - numNeigh][j - numNeigh] = currDisp                

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id