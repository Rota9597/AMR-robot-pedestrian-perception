**AMR Robot Vision-Based Pedestrian Perception** refers to the capability of Autonomous Mobile Robots (AMRs) to detect, track, and understand pedestrians in their surroundings using visual sensors. This technology enables robots to safely navigate, avoid collisions, and interact in dynamic human-populated environments (e.g., warehouses, hospitals, retail spaces). 

## Core Components
1. **Visual Sensors**
- **Cameras**: RGB, depth (RGB-D), or stereo cameras to capture environmental data.
- **Multi-Sensor Fusion** (optional): Integration with LiDAR, radar, or IMU to enhance robustness in complex scenarios.
2. **Pedestrian Detection & Recognition**
- **Object Detection**: Deep learning models (e.g., YOLO, Faster R-CNN, EfficientDet) to identify pedestrians in real time.
- **Attribute Analysis**: Estimating pose, motion direction, speed, and intent (e.g., sudden movements).
3. **Tracking & Trajectory Prediction**
- **Multi-Object Tracking (MOT)**: Algorithms like DeepSORT or SORT to maintain consistent IDs for pedestrians across frames.
- **Trajectory Forecasting**: Temporal models (LSTM, Transformer) to predict future paths and enable proactive collision avoidance.
4. **Scene Understanding**
- **Semantic Segmentation**: Mask R-CNN or similar models to segment pedestrian regions and distinguish them from obstacles.
- **Contextual Awareness**: Leveraging environmental semantics (e.g., corridors, crosswalks) to refine behavior predictions.

## Key technical components
1. **Pure Visual Perception Pipeline (RGB → 3D Understanding)**: Reconstruct 3D scene geometry and semantics purely from monocular RGB images.
2. **Monocular Depth Estimation with Self-Supervised Learning**: Train depth networks without ground-truth depth labels.
3. **Geometric-Constrained Pedestrian Detection**: Detect pedestrians with physically plausible 3D bounding boxes.
4. **Ego-Motion Aware Tracking in 2.5D Space**: Track pedestrians while compensating for the robot’s own motion.

## Applications
1. **Warehouse Logistics**
- AMRs transporting goods while avoiding workers in busy industrial settings.
2. **Service Robotics**
- Retail or hospitality robots navigating around customers without disrupting workflows.
3. **Healthcare**
- Delivery robots in hospitals adapting to the movements of staff and patients.

# Week1: Implement camera calibration toolkit (Zhang's method)

### Step 1： Checkerboard Preparation
- Checkerboard Pattern:
  - Example: An 8x6 grid (internal corners: 7x5).
  - Each square must have a precise physical size (e.g., 2 cm x 2 cm).
  - Print on a flat, rigid surface (e.g., acrylic board) to avoid warping.
- code
  ```mips
  import cv2
  pattern_size = (7, 5)  # Columns x Rows (internal corners)
  square_size = 20       # Millimeters
  img = cv2.drawChessboardCorners((800, 600), pattern_size, corners, True)
  cv2.imwrite("checkerboard.png", img)```
### Step 2：Data Acquisition
__Capture Guidelines__
- **Camera**: Use the target camera with fixed focus (disable auto-focus).
- **Number of Images**: 15–30 images (minimum 10 valid ones).
- **Requirements**:
  - Cover all regions of the frame (center, edges, corners).
  - Vary checkerboard poses (tilt, rotate, different distances).
  - Avoid motion blur, overexposure, or reflections.
### Step 3：Corner Detection
**code**
```mips
import cv2
import numpy as np

# Checkerboard parameters
pattern_size = (7, 5)  # Columns x Rows (internal corners)
square_size = 20       # Physical size per square (mm)

# Storage for 3D-2D correspondences
obj_points = []  # 3D points (Z=0)
img_points = []  # 2D image points

# Generate 3D coordinates (Z=0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# Corner detection parameters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for img_path in image_paths:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # Sub-pixel refinement
        corners_refined = cv2.cornerSubPix(
            gray, 
            corners, 
            (11, 11),  # Search window size
            (-1, -1),  # Zero zone (disabled)
            criteria
        )
        
        obj_points.append(objp)
        img_points.append(corners_refined)
        
        # Visualization (optional)
        cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
        cv2.imshow("Detected Corners", img)
        cv2.waitKey(500)
```
**Troubleshooting**
- **Failed Detection**:
  - Verify **pattern_size** matches the actual checkerboard.
  - Adjust **cv2.findChessboardCorners **flags (e.g., cv2.CALIB_CB_ADAPTIVE_THRESH).
  - Manually crop image borders to remove clutter.
  
### Step 4：Camera Calibration
**code**
```mips
# Perform calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, 
    img_points, 
    gray.shape[::-1],  # Image dimensions (width, height)
    None, 
    None,
    flags=cv2.CALIB_FIX_K3  # Use k1/k2 radial distortion model
)

print("Camera Matrix (K):\n", K)
print("Distortion Coefficients (k1, k2, p1, p2, k3):\n", dist)
```
**Parameter Details**
- **Flags**:
  - **cv2.CALIB_FIX_K3**: Fixes k3 to prevent overfitting.
  - **cv2.CALIB_ZERO_TANGENT_DIST**: Ignores tangential distortion (p1/p2).
- O**utputs**:
  - **K**: 3x3 intrinsic matrix
  $$K=
  \begin{bmatrix}
  f_x & 0 & c_x \\
  0 & f_y & c_y \\
  0 & 0 & 1
  \end{bmatrix}
  $$
  - **dist**: Distortion coefficients **[k1, k2, p1, p2, k3]**.

### Step 5： Validation
**Reprojection Error Calculation**
```mips
mean_error = 0.0
for i in range(len(obj_points)):
    img_points_reproj, _ = cv2.projectPoints(
        obj_points[i], 
        rvecs[i], 
        tvecs[i], 
        K, 
        dist
    )
    error = cv2.norm(img_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
    mean_error += error

print(f"Mean Reprojection Error: {mean_error / len(obj_points):.3f} pixels")
```
- **Acceptance Criteria**: Error < 0.5 pixels
**Undistortion Visualization**
```mips
img = cv2.imread("test_image.jpg")
h, w = img.shape[:2]

# Optimize camera matrix (optional)
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

# Undistort
undistorted = cv2.undistort(img, K, dist, None, new_K)

# Display comparison
cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)
```
### Step 6：CLI Tool Packaging
**Example Python Script**
```mips
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Zhang's Camera Calibration Toolkit")
    parser.add_argument("--image_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--pattern_size", type=str, default="7x5", help="Checkerboard inner corners (e.g., 7x5)")
    parser.add_argument("--square_size", type=float, default=20.0, help="Physical square size (mm)")
    parser.add_argument("--output", type=str, default="calibration.json", help="Output JSON file")
    args = parser.parse_args()

    # Execute calibration (steps omitted for brevity)...
    
    # Save results
    calibration_data = {
        "K": K.tolist(),
        "dist": dist.tolist(),
        "reprojection_error": mean_error,
        "image_size": [w, h]
    }
    with open(args.output, 'w') as f:
        json.dump(calibration_data, f, indent=2)

if __name__ == "__main__":
    main()
```
**Usage**
```mips
python calibrate.py --image_dir ./calib_images/ --pattern_size 7x5 --square_size 20 --output cam_params.json
```

### Step 7：Advanced Optimization
**Calibration Refinement**
1. **Outlier Rejection**:
- Remove images with reprojection errors > 2σ from the mean.
- Recalibrate with the cleaned dataset.
2. **Multi-Stage Calibration**:
- First fix principal point (cx, cy) for coarse calibration.
- Refine all parameters in a second pass.
3. **Non-Planar Targets** (Optional):
- Use ChArUco boards for improved accuracy in non-ideal conditions.
**Industrial Enhancements**
- **Automated Capture**: Use robotic arms to position the checkerboard.
- **Multi-Zone Calibration**: Calibrate across different zoom levels (for zoom lenses).
- **Temperature Compensation**: Record ambient temperature during calibration.
  
## Finding
1. Must the checkerboard be perfectly flat because any bending invalidates the 3D coordinate assumption. Use rigid materials like acrylic.
2. Reasons can cause reprojection error vary widely:
   - diverse checkerboard poses.
   - the physical square_size.
   - Disable auto-focus/optical stabilization.