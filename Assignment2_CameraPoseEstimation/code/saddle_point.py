import numpy as np
from numpy.linalg import inv, lstsq

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