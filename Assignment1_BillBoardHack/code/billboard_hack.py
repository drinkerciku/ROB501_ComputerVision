# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from matplotlib import pyplot
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image - use if you find useful.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../images/yonge_dundas_square.jpg')
    Ist = imread('../images/uoft_soldiers_tower_light.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    IstEqual = histogram_eq(Ist)

    # Compute the perspective homography we need...
    
    ## We need to compute the homography from the YDs to StG,
    ## as we can use linear interpolation to roughly determine
    ## he intensity per pixel needed on the billboard.
    H, A = dlt_homography(Iyd_pts, Ist_pts)

    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!
    
    ## We can iterate along a rectangle that contains the bounding box
    ## and check if it is contained within the ROI (Iyd_pts).
    
    ## Initialize a Path variable
    YD_ROI = Path(Iyd_pts.T)
    
    x_min = np.min(bbox[0])
    x_max = np.max(bbox[0])
    y_min = np.min(bbox[1])
    y_max = np.max(bbox[1])
    
    pointCurr = np.zeros((2,1))
    xHomoGen = np.zeros((3,1))
    xTilda = np.zeros((3,1))
    interpX = np.zeros((2,1))
    I_x = 0

    ## Process within the new bounding box 
    for j in range(y_min, y_max, 1):
        for i in range(x_min, x_max, 1):
            
            pointCurr[0][0] = i
            pointCurr[1][0] = j
            
            if YD_ROI.contains_point(pointCurr):
                
                # Perform Homography transformation to St.G
                xHomoGen[0][0] = i
                xHomoGen[1][0] = j
                xHomoGen[2][0] = 1
                xTilda = np.dot(H, xHomoGen)
                
                # Turn xTilda into inhomogenous form
                xBar = xTilda*(1/xTilda[2][0])
                
                # Run bilinear interpolation
                interpX[0][0] = xBar[0][0]
                interpX[1][0] = xBar[1][0]
                
                I_x = bilinear_interp(IstEqual, interpX)
                
                # Populate hacked image
                Ihack[j][i][0] = I_x
                Ihack[j][i][1] = I_x
                Ihack[j][i][2] = I_x
                
            else: 
                continue
            

    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!

    #------------------

    # Visualize the result, if desired...
    pyplot.imshow(Ihack)
    pyplot.show()
    # imwrite(Ihack, 'billboard_hacked.png');

    return Ihack

if __name__ == '__main__':
    
    result = billboard_hack()
