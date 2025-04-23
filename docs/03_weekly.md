# Week3: Create depth-aware data augmentation pipeline
## Core Design Principles
- **Geometric Consistency**: Ensure RGB images and depth maps are transformed synchronously to maintain pixel-level alignment.
- **Depth Validity**: Augment depth maps under physical constraints (e.g., avoid negative depths, preserve occlusion relationships).
- **Efficiency**: Support batch processing and integrate with training frameworks (e.g., PyTorch DataLoader).

## Basic Synchronized Transformations
### Step 1： Spatial Transformations (Rotation, Translation, Scaling)
**Principle**: Apply identical affine transformation matrices to RGB and depth maps.

**Code:**
```mips
import cv2
import numpy as np
import random

def apply_affine_to_pair(rgb, depth):
    # Generate random affine parameters
    angle = random.uniform(-15, 15)  # Rotation angle (degrees)
    tx = random.uniform(-0.1, 0.1) * rgb.shape[1]  # Horizontal translation (10% of width)
    ty = random.uniform(-0.1, 0.1) * rgb.shape[0]  # Vertical translation
    scale = random.uniform(0.9, 1.1)  # Scaling factor

    # Build affine matrix
    M = cv2.getRotationMatrix2D(
        center=(rgb.shape[1]//2, rgb.shape[0]//2), 
        angle=angle, 
        scale=scale
    )
    M[:, 2] += [tx, ty]  # Apply translation

    # Apply transformation
    rgb_aug = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]), 
                    flags=cv2.INTER_LINEAR)
    
    # Use nearest-neighbor interpolation for depth maps
    depth_aug = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]),
                    flags=cv2.INTER_NEAREST)
    
    return rgb_aug, depth_aug
```
### Step 2: Lighting/Contrast Adjustment
**Principle**: Modify RGB only; depth maps remain unchanged.

**Code**

```mips
def adjust_lighting(rgb, depth):
    # Random brightness/contrast adjustment
    alpha = random.uniform(0.8, 1.2)  # Contrast (1.0 = original)
    beta = random.uniform(-30, 30)    # Brightness offset
    
    rgb_aug = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
    return rgb_aug, depth.copy()
```

## Geometry-Consistent Augmentation

### Virtual Object Insertion

**Goal**: Insert 3D objects into scenes while preserving geometric validity.
**Steps**:
1. **Object Preparation**:
   - Use 3D models with textures from datasets like ShapeNet.
   - Generate object depth maps based on camera parameters.
2. Insertion Logic:
   ```mips
   def insert_virtual_object(rgb, depth, obj_rgb, obj_depth, camera_matrix):
    # Randomly select valid insertion point
    valid_y, valid_x = np.where(depth > 0)
    if not valid_x.size:
        return rgb, depth
    idx = np.random.randint(0, len(valid_x))
    center_x, center_y = valid_x[idx], valid_y[idx]
    
    # Calculate object scale based on depth
    obj_real_height = 1.0  # Physical height (meters)
    fx = camera_matrix[0, 0]
    img_height = obj_real_height * fx / depth[center_y, center_x]
    scale = img_height / obj_rgb.shape[0]
    
    # Rescale object
    obj_rgb_scaled = cv2.resize(obj_rgb, None, fx=scale, fy=scale)
    obj_depth_scaled = cv2.resize(obj_depth, None, fx=scale, fy=scale, 
                                interpolation=cv2.INTER_NEAREST)
    
    # Apply occlusion-aware blending
    h, w = obj_rgb_scaled.shape[:2]
    roi = (center_y:center_y+h, center_x:center_x+w)
    mask = obj_depth_scaled < depth[roi]
    rgb[roi][mask] = obj_rgb_scaled[mask]
    depth[roi][mask] = obj_depth_scaled[mask]
    
    return rgb, depth
   ```
### 3D-Aware Cropping
**Principle**: Crop within valid depth regions.

**Code**:
```mips
def depth_aware_crop(rgb, depth, crop_size=(256, 256)):
    # Find valid depth regions
    valid_y, valid_x = np.where(depth > 0)
    if not valid_x.size:
        return rgb, depth
    
    # Randomly select crop center
    idx = np.random.randint(0, len(valid_x))
    x, y = valid_x[idx], valid_y[idx]
    
    # Calculate crop window
    h, w = crop_size
    x1 = max(0, x - w//2)
    y1 = max(0, y - h//2)
    x2 = min(rgb.shape[1], x1 + w)
    y2 = min(rgb.shape[0], y1 + h)
    
    return rgb[y1:y2, x1:x2], depth[y1:y2, x1:x2]
```
## Occlusion Simulation

### Random Rectangular Occlusion
```mips
def simulate_occlusion(rgb, depth, max_size=100):
    h, w = rgb.shape[:2]
    occ_h = random.randint(20, max_size)
    occ_w = random.randint(20, max_size)
    x = random.randint(0, w - occ_w)
    y = random.randint(0, h - occ_h)
    
    # Apply occlusion
    rgb[y:y+occ_h, x:x+occ_w] = 0  # Black patch
    depth[y:y+occ_h, x:x+occ_w] = 0  # Zero depth
    return rgb, depth
```
### Depth-Based Dynamic Occlusion
**Principle**: Simulate near objects occluding distant ones.

**Code**:
```mips
def dynamic_occlusion(rgb, depth, camera_matrix):
    # Generate virtual occluder
    obj_z = np.random.uniform(0.5, 5.0)  # Depth (meters)
    obj_size = np.random.uniform(0.2, 1.0)  # Physical size (meters)
    fx = camera_matrix[0, 0]
    img_size = int(obj_size * fx / obj_z)  # Projected size
    
    # Random position
    x = np.random.randint(0, rgb.shape[1] - img_size)
    y = np.random.randint(0, rgb.shape[0] - img_size)
    
    # Create occlusion mask
    obj_depth = np.full((img_size, img_size), obj_z)
    background_patch = depth[y:y+img_size, x:x+img_size]
    mask = obj_depth < background_patch
    
    # Apply occlusion
    rgb[y:y+img_size, x:x+img_size][mask] = np.random.randint(0, 255, (mask.sum(), 3))
    depth[y:y+img_size, x:x+img_size][mask] = obj_z
    return rgb, depth
```

## Depth-Guided Color Augmentation

### Depth-Dependent Color Jitter
**Principle**: Apply stronger color variations to closer regions.
**Code**:
```mips
def depth_guided_color_jitter(rgb, depth):
    # Normalize depth to [0, 1]
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    
    # Per-pixel jitter strength
    strength = 0.1 + 0.3 * depth_norm  # Stronger for closer regions
    
    # Apply HSV transformations
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] += np.random.uniform(-10, 10) * strength  # Hue
    hsv[..., 1] *= np.random.uniform(0.8, 1.2) * strength  # Saturation
    hsv[..., 2] *= np.random.uniform(0.8, 1.2) * strength  # Value
    
    # Clamp values
    hsv[..., 0] = np.clip(hsv[..., 0], 0, 180)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), depth)
```

## Pipeline Integration

### Full Pipeline Implementation
```mips
class DepthAwareAugmentor:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        
    def __call__(self, rgb, depth):
        # Basic spatial transforms
        rgb, depth = apply_affine_to_pair(rgb, depth)
        
        # Lighting adjustment
        rgb, depth = adjust_lighting(rgb, depth)
        
        # Geometric augmentations
        if random.random() < 0.5:
            rgb, depth = insert_virtual_object(rgb, depth, ...)
        
        # Occlusion simulation
        if random.random() < 0.3:
            rgb, depth = simulate_occlusion(rgb, depth)
        
        # Depth-guided color jitter
        rgb, depth = depth_guided_color_jitter(rgb, depth)
        
        return rgb, depth
```

### Batch Processing with Albumentations
```mips
import albumentations as A
from albumentations.core.transforms_interface import DualTransform

class DepthAwareAffine(DualTransform):
    def apply(self, img, **params):
        return apply_affine(img)
    
    def apply_to_depth(self, depth, **params):
        return apply_affine_depth(depth)

# Define pipeline
transform = A.Compose([
    DepthAwareAffine(p=0.5),
    A.RandomBrightnessContrast(p=0.2),  # RGB-only
    A.Cutout(num_holes=3, max_h_size=30, max_w_size=30, p=0.3),
], additional_targets={'depth': 'image'})

# Usage
augmented = transform(image=rgb, depth=depth)
rgb_aug, depth_aug = augmented['image'], augmented['depth']
```

## Validation & Debugging

### Visualization
```mips
import matplotlib.pyplot as plt

def visualize(rgb, depth):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title('RGB')
    plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap='jet')
    plt.title('Depth')
    plt.show()

# Test augmentation
rgb_aug, depth_aug = augmentor(rgb_orig, depth_orig)
visualize(rgb_aug, depth_aug)
```

### Geometric Consistency Check
```mips
def verify_consistency(rgb, depth, camera_matrix):
    # Randomly sample points
    points = np.random.randint(0, min(rgb.shape[:2]), size=(10, 2))
    
    for y, x in points:
        z = depth[y, x]
        if z <= 0:
            continue
        
        # Back-project to 3D
        X = (x - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
        Y = (y - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
        
        # Re-project
        x_proj = int(X * camera_matrix[0, 0] / z + camera_matrix[0, 2])
        y_proj = int(Y * camera_matrix[1, 1] / z + camera_matrix[1, 2])
        
        assert abs(x_proj - x) < 1 and abs(y_proj - y) < 1, "Geometric inconsistency!"
```

## Optimization Tips
- **Depth Storage**: Store depth maps as **uint16** (millimeter precision), convert to float during processing.
- **Occlusion Handling**: Fill occluded depth regions with background depth + noise.
- **GPU Acceleration**:
  ```mips
  import kornia.augmentation as K

  aug = K.AugmentationSequential(
    K.RandomAffine(degrees=15, translate=0.1, scale=(0.9, 1.1)),
    K.ColorJitter(0.1, 0.1, 0.1, 0.1),
    data_keys=['input', 'mask']  # 'mask' for depth
  )

  # Apply to tensors
  rgb_tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0)
  depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
  rgb_aug, depth_aug = aug(rgb_tensor, depth_tensor)
  ```

## Application: Depth Estimation Training
```mips
class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, rgb_paths, depth_paths, transform):
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.transform = transform
        
    def __getitem__(self, idx):
        rgb = cv2.imread(self.rgb_paths[idx])[:, :, ::-1]  # BGR→RGB
        depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_ANYDEPTH)
        
        if self.transform:
            augmented = self.transform(image=rgb, depth=depth)
            rgb = augmented['image']
            depth = augmented['depth']
            
        return torch.tensor(rgb).permute(2,0,1).float(), \
               torch.tensor(depth).unsqueeze(0).float()

# Initialize DataLoader
dataset = DepthDataset(rgb_files, depth_files, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
```