import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    
    # check if points are colinear
    
    # Augment input to operate on homogenous coordinates
    weights = np.ones((1, 4))
    newI1pts = np.vstack([I1pts, weights])
    newI2pts = np.vstack([I2pts, weights])
    
    # Initialize matrix A according to input
    A = np.zeros((8,9))
    
    for i in range(0,4,1):
        
        A[2*i][0] = -newI1pts[0][i]
        A[2*i][1] = -newI1pts[1][i]
        A[2*i][2] = -newI1pts[2][i]
        A[2*i][6] = newI1pts[0][i]*newI2pts[0][i]
        A[2*i][7] = newI1pts[1][i]*newI2pts[0][i]
        A[2*i][8] = newI1pts[2][i]*newI2pts[0][i]
        
        A[2*i + 1][3] = -newI1pts[0][i]
        A[2*i + 1][4] = -newI1pts[1][i]
        A[2*i + 1][5] = -newI1pts[2][i]
        A[2*i + 1][6] = newI1pts[0][i]*newI2pts[1][i]
        A[2*i + 1][7] = newI1pts[1][i]*newI2pts[1][i]
        A[2*i + 1][8] = newI1pts[2][i]*newI2pts[1][i]
    
    hSol = null_space(A)
    hSol = hSol/hSol[8][0]
    
    H = np.reshape(hSol, (3,3))
    
    #------------------

    return H, A