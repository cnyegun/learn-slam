import numpy as np
import pandas as pd
import cv2 as cv

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os

fx, fy = 517.3, 516.5
cx, cy = 318.6, 255.3
orb = cv.ORB_create(nfeatures=2000)
bf2 = cv.BFMatcher(cv.NORM_HAMMING)  # no crossCheck for knnMatch
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.zeros(4)

# Load filenames
img_dir = 'rgbd_dataset_freiburg1_xyz/rgb/'
img_files = sorted(os.listdir(img_dir))

depth_dir = 'rgbd_dataset_freiburg1_xyz/depth/'
depth_files = sorted(os.listdir(depth_dir))

# Build timestamp lookup for depth images
depth_timestamps = np.array([float(f.replace('.png', '')) for f in depth_files])

def find_closest_depth(rgb_file):
    rgb_ts = float(rgb_file.replace('.png', ''))
    idx = np.argmin(np.abs(depth_timestamps - rgb_ts))
    return depth_files[idx]

def backproject(pts, depth_img, K):
    """Back-project 2D keypoints to 3D using depth map. Returns 3D points and valid mask."""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    pts3d = []
    valid = []
    for pt in pts:
        u, v = int(round(pt[0])), int(round(pt[1]))
        if 0 <= v < depth_img.shape[0] and 0 <= u < depth_img.shape[1]:
            z = depth_img[v, u] / 5000.0  # TUM depth scale factor
            if z > 0:
                x = (pt[0] - cx) * z / fx
                y = (pt[1] - cy) * z / fy
                pts3d.append([x, y, z])
                valid.append(True)
                continue
        pts3d.append([0, 0, 0])
        valid.append(False)
    return np.array(pts3d), np.array(valid)

# World-to-camera accumulation
R_wc = np.eye(3)
t_wc = np.zeros((3,1))
trajectory = []

for prev, curr in zip(img_files, img_files[1:]):
    img1 = cv.imread(img_dir + prev)
    img2 = cv.imread(img_dir + curr)

    # Detect features
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matches = bf2.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 6:
        trajectory.append((-R_wc.T @ t_wc).copy())
        continue

    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Back-project frame 1 keypoints to 3D using depth
    depth_file = find_closest_depth(prev)
    depth_img = cv.imread(depth_dir + depth_file, cv.IMREAD_UNCHANGED)
    pts3d, valid = backproject(pts1, depth_img, K)

    if np.sum(valid) < 6:
        trajectory.append((-R_wc.T @ t_wc).copy())
        continue

    # Keep only points with valid depth
    pts3d_valid = pts3d[valid].astype(np.float64)
    pts2_valid = pts2[valid].astype(np.float64)

    # Use PnP + RANSAC to find pose of camera 2 relative to camera 1's 3D points
    success, rvec, tvec, inliers = cv.solvePnPRansac(
        pts3d_valid, pts2_valid, K, dist_coeffs,
        iterationsCount=1000, reprojectionError=3.0,
        flags=cv.SOLVEPNP_ITERATIVE
    )

    if not success or inliers is None or len(inliers) < 6:
        trajectory.append((-R_wc.T @ t_wc).copy())
        continue

    R, _ = cv.Rodrigues(rvec)
    t = tvec

    # Accumulate world-to-camera transform
    # R, t from solvePnP: transforms 3D points from camera 1 frame to camera 2 frame
    t_wc = R @ t_wc + t
    R_wc = R @ R_wc

    # Camera position in world = -R_wc^T @ t_wc
    pos = -R_wc.T @ t_wc
    trajectory.append(pos.copy())

trajectory = np.array(trajectory).squeeze()

# Load ground truth
gt = pd.read_csv('rgbd_dataset_freiburg1_xyz/groundtruth.txt',
                 comment='#', sep=' ',
                 names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

# Interpolate ground truth to match each RGB frame timestamp
rgb_timestamps = np.array([float(f.replace('.png', '')) for f in img_files])
gt_timestamps = gt['timestamp'].values
gt_xyz = gt[['tx', 'ty', 'tz']].values

# Interpolate GT positions at each RGB timestamp (skip first since trajectory starts at frame 1)
traj_timestamps = rgb_timestamps[1:]  # trajectory has len(img_files)-1 entries
gt_interp = np.column_stack([
    np.interp(traj_timestamps, gt_timestamps, gt_xyz[:, i]) for i in range(3)
])

def umeyama_alignment(est, gt):
    """Align estimated trajectory to ground truth using Umeyama method (rotation + scale + translation)."""
    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    est_c = est - mu_est
    gt_c = gt - mu_gt

    # Covariance
    H = est_c.T @ gt_c / len(est)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T

    # Scale
    var_est = np.sum(est_c ** 2) / len(est)
    scale = np.sum(S * np.diag(D)) / var_est

    # Translation
    t = mu_gt - scale * R @ mu_est
    return R, t, scale

# Use all 3 dimensions for alignment
est_3d = trajectory  # (N, 3) in camera 0 frame
gt_3d = gt_interp    # (N, 3) in mocap frame

R_align, t_align, s_align = umeyama_alignment(est_3d, gt_3d)
est_aligned = (s_align * (R_align @ est_3d.T).T + t_align)

# Plot XZ (top-down view) of aligned estimate vs ground truth
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(est_aligned[:, 0], est_aligned[:, 2])
ax1.set_title('Estimated (aligned to GT)')
ax1.set_aspect('equal')
ax2.plot(gt_3d[:, 0], gt_3d[:, 2])
ax2.set_title('Ground truth')
ax2.set_aspect('equal')
plt.tight_layout()
plt.show()
