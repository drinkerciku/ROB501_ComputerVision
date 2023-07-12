import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# -------------- HELPER FUNCTIONS -------------- #
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


def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
    
    # The input patch is the PxP neighbourhood of pixels after computing 
    # X-junctions and we need only perform a least-square minimization 
    # over all pixels for the parameters of the quadratic fitting.
    # The patch is assumed to be have been already blurred.
    
    # Initialize the bounding 
    pt = np.array([0.0, 0.0], dtype = np.float64)
    
    # The least-squares problem determines the x that minimizes || b - ax ||
    # Construct matrices b and A.
    dimX = I.shape[1]
    dimY = I.shape[0]
    
    # PxP rows and 6 columns for constants. 
    A = np.zeros((dimX*dimY, 6), dtype = np.float64)
    # PxP rows and 1 column for pixel intensities
    b = np.zeros((dimX*dimY, 1), dtype = np.float64)
    
    A_k = lambda x, y : np.array([x*x, x*y, y*y, x, y, 1.0], dtype = np.float64)
    
    for j in range(0, dimY, 1):
        for i in range(0, dimX, 1):
            
            A[j*dimX + i][:] = A_k (i, j)
            b[j*dimX + i][0] = np.float64(I[j][i]) 
            
    # Compute the least squares result
    coeff = lstsq(A, b, rcond = None)
    
    # Construct the coefficent matrices to compute the saddle subpixeel accuracy 
    # point.
    
    coeffA = np.zeros((2, 2), dtype = np.float64)
    coeffB = np.zeros((2,1), dtype = np.float64)
    
    coeffA[0][0] = 2*coeff[0][0]
    coeffA[0][1] = coeff[0][1]
    coeffA[1][0] = coeff[0][1]
    coeffA[1][1] = 2*coeff[0][2]
    coeffB[0][0] = coeff[0][3]
    coeffB[1][0] = coeff[0][4]
    
    # compute pt
    pt = (-1.0)*np.dot(inv(coeffA), coeffB)
    
    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt

# ---------------------------------------------- #

def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---

    # Wpts contains the coordinates of every junction in (x, y, 0.0f) in meters.
    # We require a mapping from Wpts -> image plane. In chapter 11.1 (pg. 686) of Szeliski,
    # a valid calibration technique can be obtained by waving a planar calibration
    # pattern in front of the camera -> the approach consists of computing a separate
    # homography transformation per image between the plane's calibration points 
    # in 3D sapce to the image plane - not a very accurate method but it is easy to implement.
    
    # We need to approximate the bounding box in the world plane.
    # Based on visual inspections, the 'near' corner junctions are 1/3 of the square's
    # edge from the board's left/right edge and ~1/5 from the top/bottom edge - these 
    # are the average values observed across the sample images provied
    
    # Since there are some inaccuracies on the perceived dimensions 
    # of squares due to varying perspective, we will use the delta 
    # from one Wpt to another to estimate teh distances from the edges.
    
    deltaSq = Wpts[0][1]
    
    dX_edge = 1/3*deltaSq + deltaSq
    dY_edge = 1/5*deltaSq + deltaSq
    
    worldTL = np.array([[Wpts[0][0] - dX_edge],[Wpts[1][0] - dY_edge]])
    worldTR = np.array([[Wpts[0][7] + dX_edge],[Wpts[1][7] - dY_edge]])
    worldBL = np.array([[Wpts[0][40] - dX_edge],[Wpts[1][40] + dY_edge]])
    worldBR = np.array([[Wpts[0][47] + dX_edge],[Wpts[1][47] + dY_edge]])
    
    worldPoly = np.hstack((worldTL, worldTR, worldBR, worldBL))
    
    # compute homography from world to image
    wrldH, wrldA = dlt_homography(worldPoly, bpoly)
    
    # initialize output array
    Ipts = np.zeros((2, Wpts.shape[1]))
    
    # helper variables
    currLandM = np.ones((3,1))
    imageLandM = np.zeros((3,1))
    inSaddleX = 0
    inSaddleY = 0
    outSaddle = np.zeros((2,1))
    
    # neighbourhood for saddle point computation - determined through tests
    omegaP = 10
    
    # process each landmark 
    for landM in range(0, Wpts.shape[1], 1):
        
        # Use homography to map on the image plane
        currLandM[0][0] = Wpts[0][landM]
        currLandM[1][0] = Wpts[1][landM]
        imageLandM = np.dot(wrldH, currLandM)
        # Safety check - converting to inhomogenous form
        inSaddleX = int(imageLandM[0][0]/imageLandM[2][0])
        inSaddleY = int(imageLandM[1][0]/imageLandM[2][0])
        
        # Solve minimization problem        
        patch = I[inSaddleY - omegaP : inSaddleY + omegaP + 1,\
                  inSaddleX - omegaP : inSaddleX + omegaP + 1]
        outSaddle = saddle_point(patch)

        # save in output buffer
        Ipts[0][landM] = outSaddle[0][0] + inSaddleX - omegaP
        Ipts[1][landM] = outSaddle[1][0] + inSaddleY - omegaP

    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts