import numpy as np
import time
import matplotlib.pyplot as plt
from ibvs_controller import ibvs_controller
from ibvs_simulation import ibvs_simulation
from dcm_from_rpy import dcm_from_rpy

import matplotlib.lines as mlines

blue_star = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=5, label='Converged')
red_square = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                          markersize=5, label='Singularity')
purple_triangle = mlines.Line2D([], [], color='blue', marker='^', linestyle='None',
                          markersize=5, label='Max Iterations Hit')

# Camera intrinsics matrix - known.
K = np.array([[500.0, 0, 400.0], 
              [0, 500.0, 300.0], 
              [0,     0,     1]])

# Target points (in target/object frame).
pts = np.array([[-0.75,  0.75, -0.75,  0.75],
                [-0.50, -0.50,  0.50,  0.50],
                [ 0.00,  0.00,  0.00,  0.00]])

# Camera poses, last and first.
C_last = np.eye(3)
t_last = np.array([[ 0.0, 0.0, -4.0]]).T
C_init = dcm_from_rpy([np.pi/5, -np.pi/4, np.pi/2])
t_init = np.array([[-0.2, 0.3, -5.0]]).T

Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))

gain = 1.6

# Sanity check the controller output if desired.
# ...

# Run simulation - use known depths.
gains = np.linspace(0.05, 2.05, num = 200)
gain_converged = [[],[]]
gain_Nconverged = [[],[]]
gain_failure = [[],[]]
result = [0, 1001]

for i in range(0, gains.shape[0], 1):
    initT = time.perf_counter()
    status = 'F'
    iterNum = 0
    
    try:
        iterNum, status = ibvs_simulation(Twc_init, Twc_last, pts, K, gains[i], True, False)
    except:
        print('Singularity')
        
    plt.figure(1)
        
    if status == 'F':
        gain_failure[0].append(gains[i])
        gain_failure[1].append(iterNum)
        plt.plot(gains[i], iterNum, 'rx')
    elif status == 'N':
        gain_Nconverged[0].append(gains[i])
        gain_Nconverged[1].append(iterNum)
        plt.plot(gains[i], iterNum, 'bo')
    else:
        gain_converged[0].append(gains[i])
        gain_converged[1].append(iterNum) 
        plt.plot(gains[i], iterNum, 'go')
        
        if iterNum < result[1]:
            result[1] = iterNum
            result[0] = gains[i]
    
    plt.xlim([0, 2.2])
    plt.ylim([-2, 200])
    plt.grid(True)
    plt.show(block = False)
    
    print("Time Elapsed {}s".format(time.perf_counter() - initT))

plt.legend(handles=[blue_star, red_square, purple_triangle])
plt.xlabel('Control Gain')
plt.ylabel('Iteration Count')
plt.show()
print('done')
print(result)