"""
SLAM Step 1: Feature Detection & Matching
==========================================
This script loads two frames from the TUM fr1/xyz dataset,
detects ORB features, matches them, and visualizes the result.

Run: python slam_step1_features.py <path_to_dataset_folder>
Example: python slam_step1_features.py rgbd_dataset_freiburg1_xyz
"""

import cv2
import numpy as np
import sys
import os


def load_rgb_filenames(dataset_path):
    """
    Parse rgb.txt to get sorted list of (timestamp, filepath) pairs.
    The file has 3 comment lines starting with '#', then:
        timestamp filename
        1305031102.175304 rgb/1305031102.175304.png
    """
    rgb_txt = os.path.join(dataset_path, "rgb.txt")
    frames = []
    with open(rgb_txt, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                timestamp = float(parts[0])
                filepath = os.path.join(dataset_path, parts[1])
                frames.append((timestamp, filepath))
    frames.sort(key=lambda x: x[0])  # sort by timestamp
    return frames


def detect_and_match(img1, img2, max_features=1000):
    """
    Detect ORB features in both images and match them.
    
    ORB = Oriented FAST and Rotated BRIEF
    - FAST: detects corners (features) quickly
    - BRIEF: computes a binary descriptor (fingerprint) for each corner
    - "Oriented" and "Rotated" make it robust to rotation
    
    Returns matched keypoints in both images.
    """
    # Step 1: Create ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)

    # Step 2: Detect keypoints and compute descriptors
    # keypoints = list of (x,y) locations where features were found
    # descriptors = matrix where each row is a 256-bit fingerprint
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    print(f"  Frame 1: {len(kp1)} features detected")
    print(f"  Frame 2: {len(kp2)} features detected")

    if des1 is None or des2 is None:
        print("  No descriptors found!")
        return [], [], []

    # Step 3: Match descriptors using Brute Force matcher
    # NORM_HAMMING because ORB descriptors are binary (compare with XOR)
    # crossCheck=True means: A matches B AND B matches A (more reliable)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Step 4: Sort by distance (lower = better match)
    matches = sorted(matches, key=lambda m: m.distance)

    print(f"  Matches found: {len(matches)}")

    # Step 5: Keep only the best matches (top 50%)
    good_matches = matches[: len(matches) // 2]
    print(f"  Good matches (top 50%): {len(good_matches)}")

    return kp1, kp2, good_matches


def visualize_features(img1, kp1, title="Features"):
    """Draw detected features on an image."""
    vis = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(title, vis)


def visualize_matches(img1, img2, kp1, kp2, matches):
    """
    Draw matches between two images side by side.
    Green lines connect matching features.
    """
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Feature Matches", vis)


def visualize_flow(img2, kp1, kp2, matches):
    """
    Draw motion vectors on frame 2.
    Each arrow shows how a feature moved from frame 1 to frame 2.
    This gives you an intuitive feel for the camera motion.
    """
    vis = img2.copy()
    for m in matches:
        # Get pixel coords of matched features
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))  # position in frame 1
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))  # position in frame 2

        # Draw arrow from old position to new position
        cv2.arrowedLine(vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)
        cv2.circle(vis, pt2, 3, (0, 0, 255), -1)

    cv2.imshow("Optical Flow (how features moved)", vis)


def main():
    if len(sys.argv) < 2:
        print("Usage: python slam_step1_features.py <path_to_dataset>")
        print("Example: python slam_step1_features.py rgbd_dataset_freiburg1_xyz")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Load frame list
    frames = load_rgb_filenames(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"Total frames: {len(frames)}")

    # Pick two frames: first frame and one ~10 frames later
    # (consecutive frames are almost identical, skip a few for visible motion)
    idx1, idx2 = 0, 10
    
    print(f"\nLoading frame {idx1} and frame {idx2}...")
    img1 = cv2.imread(frames[idx1][1])
    img2 = cv2.imread(frames[idx2][1])

    if img1 is None or img2 is None:
        print(f"Error: could not load images")
        print(f"  Tried: {frames[idx1][1]}")
        print(f"  Tried: {frames[idx2][1]}")
        sys.exit(1)

    print(f"  Image size: {img1.shape[1]}x{img1.shape[0]}")

    # Convert to grayscale (ORB works on grayscale)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect and match features
    print("\nDetecting and matching features...")
    kp1, kp2, good_matches = detect_and_match(gray1, gray2)

    if len(good_matches) == 0:
        print("No matches found!")
        sys.exit(1)

    # Extract matched point coordinates for later use (pose estimation)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Show average pixel displacement
    displacement = np.mean(np.linalg.norm(pts2 - pts1, axis=1))
    print(f"\n  Average feature displacement: {displacement:.1f} pixels")
    print(f"  This tells you roughly how much the camera moved between frames.")

    # Visualize
    print("\nShowing visualizations (press any key to cycle, 'q' to quit)...")

    visualize_features(img1, kp1, "Frame 1 - Detected Features")
    print("  [Window 1] Detected features in frame 1. Press any key...")
    cv2.waitKey(0)

    visualize_matches(img1, img2, kp1, kp2, good_matches)
    print("  [Window 2] Feature matches between frames. Press any key...")
    cv2.waitKey(0)

    visualize_flow(img2, kp1, kp2, good_matches)
    print("  [Window 3] Optical flow arrows. Press any key...")
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # ---------------------------------------------------------------
    # WHAT'S NEXT (Step 2): 
    # We now have matched points pts1 and pts2.
    # Next step: use these to compute the Essential matrix E,
    # decompose it into R and t, and triangulate 3D points.
    # ---------------------------------------------------------------
    print("\n--- Next step ---")
    print(f"We have {len(good_matches)} matched point pairs.")
    print("These go into cv2.findEssentialMat() to recover camera motion.")


if __name__ == "__main__":
    main()
