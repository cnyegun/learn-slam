import numpy as np
import pandas as pd
import cv2 as cv

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os 

fx, fy = 517.3, 516.5
cx, cy = 318.6, 255.3
orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf2 = cv.BFMatcher(cv.NORM_HAMMING)  # no crossCheck
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Load filenames
img_dir = 'rgbd_dataset_freiburg1_xyz/rgb/'
img_files = sorted(os.listdir(img_dir))

Rs, ts = [], []
R_global = np.eye(3)
t_global = np.zeros((3,1))
trajectory = []

for prev, curr in zip(img_files[::3], img_files[3::3]):
    img1 = cv.imread(img_dir + prev)
    img2 = cv.imread(img_dir + curr)

    # Detect features
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matches = bf2.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Extract matched points
    D = []
    for m in good_matches:
        D.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))
    pts1 = np.float32([pt1 for pt1, _ in D])
    pts2 = np.float32([pt2 for _, pt2 in D])


    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC)
    _, R, t, mask2 = cv.recoverPose(E, pts1, pts2, K, mask=mask)

    # Accumulate Rs and ts
    Rs.append(R)
    ts.append(t)

    # Accumulate to global R and global t
    t_global = t_global + R_global @ t
    R_global = R_global @ R
    trajectory.append(t_global.copy())


    #print(f"Rotation angle {np.degrees(np.arccos((np.trace(R) - 1) / 2)):.2f}°\n")
    #print(f"Camera direction: \nX: {t[0].item():.4f}, Y: {t[1].item():.4f}, Z: {t[2].item():.4f}\n") 

    # Calculate projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))

    # Triangulate to get the points in 3D coord
    points4D = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points3D = points4D[:3] / points4D[3]

    #print(points3D)

trajectory = np.array(trajectory).squeeze()
gt = pd.read_csv('rgbd_dataset_freiburg1_xyz/groundtruth.txt',
                 comment='#', sep=' ',
                 names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

traj_norm = trajectory - trajectory[0]
gt_norm = gt[['tx', 'tz']].values - gt[['tx', 'tz']].values[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(traj_norm[:, 0], traj_norm[:, 2])
ax1.set_title('Estimated')
ax1.set_xlabel('X')
ax1.set_ylabel('Z')

ax2.plot(gt_norm[:, 0], gt_norm[:, 1])
ax2.set_title('Ground truth')
ax2.set_xlabel('X')
ax2.set_ylabel('Z')

plt.tight_layout()
plt.show()
