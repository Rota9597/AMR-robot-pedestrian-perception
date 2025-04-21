# 👁️ AMR Robot Vision-Based Pedestrian Perception  
*Monocular camera perception system for autonomous mobile robots in human-centric environments*

## 🎯 Project Description  
This project develops a vision-centric perception stack for Autonomous Mobile Robots (AMRs) using **monocular depth estimation** and geometric reasoning. Key technical components include:  
- 📷 Pure visual perception pipeline (RGB ➔ 3D understanding)  
- 📐 Monocular depth estimation with self-supervised learning  
- 👥 Geometric-constrained pedestrian detection  
- 🛤️ Ego-motion aware tracking in 2.5D space  

Students will explore:  
- Depth ambiguity resolution in monocular vision  
- Occlusion reasoning through temporal fusion  
- Computational efficiency optimization for embedded GPUs  
- Ethical visual perception design (privacy-preserving blurring)

## 🔧 Skill Requirements  
### Core Competencies  
- **Computer Vision**: OpenCV, PyTorch, image geometry  
- **Depth Estimation**: Familiarity with frameworks like MiDaS/DepthAnything  
- **Robotics Basics**: ROS navigation stack, TF transformations  
- **Linear Algebra**: Camera matrices, epipolar geometry  

### Specialized Skills  
- 📦 Experience with synthetic datasets (CARLA/Unity Perception)  
- 🧠 Understanding self-supervised learning paradigms  
- ⏱️ Real-time system optimization (TensorRT/ONNX)  
- 🤖 Basic knowledge of Visual SLAM (ORB-SLAM3)  

## 🗓️ Milestones  
| Week | Phase | Objectives |  
|------|-------|------------|  
| 1-2  | 📸 **Data Pipeline** | Build mixed real/synthetic dataset with depth ground truth |  
| 3-5  | 🎚️ **Depth Network** | Implement lightweight monocular depth estimator (e.g., MobileDepth) |  
| 6-8  | 👤 **Pedestrian Detection** | Develop geometry-aware YOLO variant (projective constraints) |  
| 9-11 | 🔄 **Temporal Fusion** | Create Kalman filter with depth uncertainty modeling |  
| 12-14| 🚦 **Safety System** | Build speed-adaptive safety bubble generator |  

## 📋 Work Breakdown Structure  
### Phase 1: Visual Perception Foundation  
1. Implement camera calibration toolkit (Zhang's method)  
2. Develop synthetic-to-real domain adaptation module  
3. Create depth-aware data augmentation pipeline  

### Phase 2: Monocular Depth Estimation  
1. Benchmark depth networks (AdaBins vs DepthFormer)  
2. Develop scale-aware loss function  
3. Implement temporal consistency constraints  

### Phase 3: Geometry-Based Detection  
1. Create perspective-navigation map converter  
2. Develop height-prior boxes for pedestrian detection  
3. Build depth-validated detection scoring system  

### Phase 4: Dynamic Tracking  
1. Implement 2.5D Kalman filter (x,y,scale)  
2. Develop occlusion reasoning with ray casting  
3. Create interaction-aware trajectory prediction  

### Phase 5: System Integration  
1. Build ROS perception node with latency monitor  
2. Develop adaptive image resolution controller  
3. Implement GDPR-compliant face blurring  

## 🚀 Deliverables  
- Real-time perception pipeline (ROS package)  
- Novel monocular depth estimation model  
- Geometry-constrained detection toolkit  
- Ethics-by-design implementation guide  
- Benchmark dataset with privacy masks  

## ⏳ Time Commitment  
- 12-14 hrs/week (include GPU cluster usage)  
- Weekly Depth Estimation Study Group  
- Bi-weekly safety review meetings  
- Final demonstration with obstacle course  

📌 *Path Specialization:  
- CV students: Depth network architecture innovation  
- Robotics students: Real-time system integration  
- Ethics-focused students: Privacy preservation modules*  

🔧 *Hardware Kit:  
- Jetson Xavier + RealSense D455  
- Custom mobile robot platform  
- Hospital corridor simulation environment*  

## Workload:

### Week1: Run fastestdet with your own camera (like laptop camera) on your local machine. https://github.com/dog-qiuqiu/FastestDet
