import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_depth_finder(K, pts_obs, pts_prev, zs_guess, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    zs_guess - nx0 np.array, points depth values (estimated guess).
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """

    #--- FILL ME IN ---
    
    # loop parameters
    N = pts_obs.shape[1]
    # initialize return value
    zs_est = np.zeros(N)
    # initialize variables
    J = np.zeros((2*N, 6))
    # reshape points
    prevCol = np.reshape(pts_prev, (2*N, 1), 'F')
    obsCol = np.reshape(pts_obs, (2*N, 1), 'F')
    # loop variables
    J_i = np.zeros((2, 6))
    iPt = np.zeros((2, 1))
    
    for i in range(0, N, 1):
        
        # compute Jacobian at current location
        iPt[0][0] = obsCol[2*i][0]
        iPt[1][0] = obsCol[2*i + 1][0]
        J_i = ibvs_jacobian(K, iPt, 1)
        J[2*i : 2*i + 2, :] = J_i
        
    # extract control components
    v_t = np.reshape(v_cam[0:3, 0], (3, 1))
    v_w = np.reshape(v_cam[3:, 0], (3, 1))
    
    # compute estimates of depth
    A = np.zeros((2*N, N))
    
    for i in range(0, N, 1):
        iPt = np.dot(J[2*i : 2*i + 2, 0 : 3], v_t)
        A[2*i][i] = iPt[0][0]
        A[2*i + 1][i] = iPt[1][0]
    
    b = (obsCol - prevCol) - np.dot(J[:, 3:], v_w)
    theta = np.linalg.lstsq(A, b, rcond = None)
    
    for i in range(0, N, 1):
        zs_est[i] = 1/theta[0][i] 
    #------------------

    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (N,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est