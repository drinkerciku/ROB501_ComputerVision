import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """

    #--- FILL ME IN ---
    
    # initialie output to default values
    J = np.zeros((2,6))
    
    # coordinates of point with respect to the principal point
    u = pt[0][0] - K[0][2]
    v = pt[1][0] - K[1][2]
    # focal length (assuming that ro = ro_h = ro_v = 1.0)
    f = K[0][0]
    
    
    # compute Jacobain
    J[0, :] = [-f/z, 0, u/z, u*v/f, -(f**2 + u**2)/f, v]
    J[1, :] = [0, -f/z, v/z, (f**2 + v**2)/f, -u*v/f, -u]
    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J