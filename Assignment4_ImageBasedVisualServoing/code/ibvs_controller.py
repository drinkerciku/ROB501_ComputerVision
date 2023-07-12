import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_controller(K, pts_des, pts_obs, zs, gain):
    """
    A simple proportional controller for IBVS.

    Implementation of a simple proportional controller for image-based
    visual servoing. The error is the difference between the desired and
    observed image plane points. Note that the number of points, n, may
    be greater than three. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K       - 3x3 np.array, camera intrinsic calibration matrix.
    pts_des - 2xn np.array, desired (target) image plane points.
    pts_obs - 2xn np.array, observed (current) image plane points.
    zs      - nx0 np.array, points depth values (may be estimated).
    gain    - Controller gain (lambda).

    Returns:
    --------
    v  - 6x1 np.array, desired tx, ty, tz, wx, wy, wz camera velocities.
    """
    v = np.zeros((6, 1))

    #--- FILL ME IN ---
    
    # loop parameters
    N = pts_des.shape[1]
    # initialize variables
    J = np.zeros((2*N, 6))
    # reshape points
    destCol = np.reshape(pts_des, (2*N, 1), 'F')
    obsCol = np.reshape(pts_obs, (2*N, 1), 'F')
    # loop variables
    J_i = np.zeros((2, 6))
    iPt = np.zeros((2, 1))
    
    # compute the stacked Jacobian
    for i in range(0, N, 1):
        
        # compute Jacobian at current location
        iPt[0][0] = obsCol[2*i][0]
        iPt[1][0] = obsCol[2*i + 1][0]
        J_i = ibvs_jacobian(K, iPt, zs[i])
        J[2*i:2*i+2, :] = J_i

    # compute the pseudo inverse of J and controller
    # output for linear and angular velocities
    JpInv = np.dot(inv(np.dot(J.T, J)), J.T)
    v = gain*np.dot(JpInv, destCol - obsCol)
    
    # ------------------

    correct = isinstance(v, np.ndarray) and \
        v.dtype == np.float64 and v.shape == (6, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return v