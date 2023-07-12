import numpy as np
from numpy.linalg import inv 

# --------------- COPY OF SUPPORT FUNCTIONS --------------- #

def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

# --------------------------------------------------------- #
def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """
    #--- FILL ME IN ---
        
    # In ROB301 we derived the inverse of the homogenous transformation
    # in the context of DH parameters as part of an SO(2) -> 
    # A_inv = [[R.T, -R.T*t], [0.T, 1]]. Now we can evaluate 
    # Jacobian's by taking a partial derivatives for the variables 
    # we will be calibrating. Keep in mind that our transformation consists of 
    # (1/z)*(K*R*x_in + R*t) -> we will be using the quotient rule.
    
    # Compute perspective projection matrix
    K_tilde = np.hstack((K, np.array([[0],[0],[0]])))
    K_tilde = np.vstack((K_tilde, np.array([0, 0, 0, 1])))
    T_cw = inv(Twc)    
    P_tilde = np.dot(K_tilde, T_cw)
    
    # In homogenous world coordinate
    Wpt_bar = np.vstack((Wpt, np.array([1])))
    # In the image plane (homogenous point)
    Wpt_image = np.dot(P_tilde, Wpt_bar)
    
    ## COMPUTATION OF THE JACOBIAN ##
    
    # Initialize the empty Jacobian
    J = np.zeros((2, 6), dtype = np.float64)
    
    # Important Variables and definitions
    R_wc = Twc[ : 3, : 3]
    R_cw = Twc[ : 3, : 3].T
    dTrans = Wpt - np.reshape(Twc[:3,3], (3,1))
    # Retrieve the rotation angles (roll, pitch, yaw)
    q_angles = rpy_from_dcm(R_wc)
    
    zPrime_Trans = lambda idT : (idT == 0)*(-R_cw[2][0]) + (idT == 1)*(-R_cw[2][1]) \
                                + (idT == 2)*(-R_cw[2][2])
                                
    zPrime_Rot = lambda delta, rotM : np.dot(np.reshape(rotM[2, :], (1,3)), delta) 

    # Helper Variables 
    Jt_i = np.zeros((4,1), dtype = np.float64)
    Jr_i = np.zeros((4,1), dtype = np.float64)
    dT_cw = np.zeros((4,4), dtype = np.float64)
    dT_cw[3][3] = 1.0
    
    # ------------------------ TRANSLATION ------------------------ # 

    # dt_x
    dT_cw[:3,3] = np.dot(-R_cw, np.array([1, 0, 0]))
    Jt_i = np.dot(np.dot(K_tilde, dT_cw), Wpt_bar)
    
    J[0][0] = Jt_i[0][0]/Wpt_image[2][0] - zPrime_Trans(0)*Wpt_image[0][0]/np.square(Wpt_image[2][0])
    J[1][0] = Jt_i[1][0]/Wpt_image[2][0] - zPrime_Trans(0)*Wpt_image[1][0]/np.square(Wpt_image[2][0])
    
    # dt_y
    dT_cw[:3,3] = np.dot(-R_cw, np.array([0, 1, 0]))
    Jt_i = np.dot(np.dot(K_tilde, dT_cw), Wpt_bar)
    
    J[0][1] = Jt_i[0][0]/Wpt_image[2][0] - zPrime_Trans(1)*Wpt_image[0][0]/np.square(Wpt_image[2][0])
    J[1][1] = Jt_i[1][0]/Wpt_image[2][0] - zPrime_Trans(1)*Wpt_image[1][0]/np.square(Wpt_image[2][0])
    
    # dt_z
    dT_cw[:3,3] = np.dot(-R_cw, np.array([0, 0, 1]))
    Jt_i = np.dot(np.dot(K_tilde, dT_cw), Wpt_bar)
    
    J[0][2] = Jt_i[0][0]/Wpt_image[2][0] - zPrime_Trans(2)*Wpt_image[0][0]/np.square(Wpt_image[2][0])
    J[1][2] = Jt_i[1][0]/Wpt_image[2][0] - zPrime_Trans(2)*Wpt_image[1][0]/np.square(Wpt_image[2][0])
    
    # ------------------------------------------------------------- #
    
    # ------------------------- ROTATION ------------------------- #
    
    # dRoll
    # obtain dR/dRoll matrix
    
    dRot = getRotDelta(0, -q_angles[0], -q_angles[1], -q_angles[2])
    dZet = zPrime_Rot(dTrans, dRot)    
    # compute the i-th rotational Jacobian
    Jr_i = np.dot(K,np.dot(dRot, dTrans)) 
    
    J[0][3] = -Jr_i[0][0]/Wpt_image[2][0] + dZet*Wpt_image[0][0]/np.square(Wpt_image[2][0])
    J[1][3] = -Jr_i[1][0]/Wpt_image[2][0] + dZet*Wpt_image[1][0]/np.square(Wpt_image[2][0])
    
    # dRoll
    # obtain dR/dRoll matrix
    dRot = getRotDelta(1, -q_angles[0], -q_angles[1], -q_angles[2])
    dZet = zPrime_Rot(dTrans, dRot)    
    # compute the i-th rotational Jacobian
    Jr_i = np.dot(K,np.dot(dRot, dTrans)) 
    
    J[0][4] = -Jr_i[0][0]/Wpt_image[2][0] + dZet*Wpt_image[0][0]/np.square(Wpt_image[2][0])
    J[1][4] = -Jr_i[1][0]/Wpt_image[2][0] + dZet*Wpt_image[1][0]/np.square(Wpt_image[2][0])
    
    # dRoll
    # obtain dR/dRoll matrix
    dRot = getRotDelta(2, -q_angles[0], -q_angles[1], -q_angles[2])
    dZet = zPrime_Rot(dTrans, dRot)    
    # compute the i-th rotational Jacobian
    Jr_i = np.dot(K,np.dot(dRot, dTrans)) 
    
    J[0][5] = -Jr_i[0][0]/Wpt_image[2][0] + dZet*Wpt_image[0][0]/np.square(Wpt_image[2][0])
    J[1][5] = -Jr_i[1][0]/Wpt_image[2][0] + dZet*Wpt_image[1][0]/np.square(Wpt_image[2][0])
    
    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J

# --------------------------- HELPER FUNCTIONS --------------------------- #

def getRotDelta(axisDelta, r, p, y):
    
    # initialize the resulting matrix
    dR = np.zeros((3,3), dtype = np.float64)
    
    cr = np.cos(r).item()
    sr = np.sin(r).item()
    cp = np.cos(p).item()
    sp = np.sin(p).item()
    cy = np.cos(y).item()
    sy = np.sin(y).item()
    
    if axisDelta == 0:
        
        rR = np.array([[0, 0, 0],[0, -sr, -cr],[0, cr, -sr]])
        rP = np.array([[cp, 0, sp],[0, 1, 0],[-sp, 0, cp]])
        rY = np.array([[cy, -sy, 0],[sy, cy, 0],[0, 0, 1]])
        
    elif axisDelta == 1:
        
        rR = np.array([[1, 0, 0],[0, cr, -sr],[0, sr, cr]])
        rP = np.array([[-sp, 0, cp],[0, 0, 0],[-cp, 0, -sp]])
        rY = np.array([[cy, -sy, 0],[sy, cy, 0],[0, 0, 1]])
    
    elif axisDelta == 2:
        
        rR = np.array([[1, 0, 0],[0, cr, -sr],[0, sr, cr]])
        rP = np.array([[cp, 0, sp],[0, 1, 0],[-sp, 0, cp]])
        rY = np.array([[-sy, -cy, 0],[cy, -sy, 0],[0, 0, 0]])
    
    else: 
        raise TypeError("Invalid axis in getRotDelta!")
    
    dR = np.dot(rR, np.dot(rP,rY))
    
    return dR