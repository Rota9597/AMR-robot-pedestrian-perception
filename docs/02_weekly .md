# Week2: Develop synthetic-to-real domain adaptation module

## Step 1：Problem Analysis & Data Preparation
**Core Challenges**
- **Domain Shift**: Distribution discrepancies between synthetic (source) and real (target) data in texture, lighting, and object details.
- **Label Scarcity**: Real-world data often lacks annotations (e.g., pixel-level labels for semantic segmentation).
  
**Data Requirements**
| Data Type | Example Datasets | Purpose |
|-----|-----|-----|
| Synthetic | GTA5, CARLA, Blender-generated | Source domain (fully annotated) |
| Real| Cityscapes, KITTI, custom captures | 	Target domain (unlabeled/sparse labels) |
**Alignment Tips**:
- Ensure semantic class consistency (e.g., match "road" or "car" categories).
- Resize synthetic data to match real data resolution (e.g., 1280×720 → 640×320).

## Step 2：Method Selection & Pipeline Design
__Approach Comparison__
| Method | Example Algorithms | Pros |Cons|
|-----|-----|-----|-----|
| **Adversarial Training** | 	DANN, ADVENT | 	No target labels required |Training instability|
| **Image-to-Image Translation**| CycleGAN, DRIT++ | 	High visual realism |	May distort semantic structures|
|**Self-Training**|Noisy Student|Leverages pseudo-labels|Depends on initial model quality|
|**Feature Alignment**|MMD, CORAL|Lightweight computation|Limited for complex domain gaps|
**Recommended Pipeline**
1. Data Preprocessing → Align color/resolution distributions
2. Image Translation → Convert synthetic data to "real style" via CycleGAN
3. Adversarial Training → Align feature spaces (source vs. target)
4. Self-Training → Refine model with pseudo-labels iteratively

## Step 3：Core Module Implementation
### Module 1: CycleGAN-Based Image Translation
**Goal**: Convert synthetic images to real-world style while preserving semantics.
**Code Implementation**:
```mips
# PyTorch Simplified Implementation
import torch
import torch.nn as nn

# Generator (U-Net architecture)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # Input: RGB (3 channels)
            nn.LeakyReLU(0.2),
            # ... add more downsampling layers
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            # ... add more upsampling layers
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.upsample(x)
        return torch.tanh(x)  # Output: [-1, 1]

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # ... add more layers
            nn.Conv2d(512, 1, 4, 1, 0)  # Output: real/fake probability
        )

    def forward(self, x):
        return self.model(x)

# Loss Functions
criterion_gan = nn.MSELoss()        # GAN loss
criterion_cycle = nn.L1Loss()       # Cycle-consistency loss
criterion_identity = nn.L1Loss()    # Identity loss

# Training Loop
for epoch in range(num_epochs):
    for real_A, real_B in zip(real_loader, synth_loader):
        # Forward pass
        fake_B = generator_B(real_A)  # Translate real→synthetic style
        rec_A = generator_A(fake_B)   # Reconstruct real image
        idt_A = generator_A(real_A)   # Identity mapping
        
        # Discriminator loss
        pred_real = discriminator_A(real_A)
        loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real))
        pred_fake = discriminator_A(fake_B.detach())
        loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        # Generator loss
        loss_G_gan = criterion_gan(discriminator_A(fake_B), torch.ones_like(pred_fake))
        loss_G_cycle = criterion_cycle(rec_A, real_A)
        loss_G_idt = criterion_identity(idt_A, real_A)
        loss_G = loss_G_gan + 10 * loss_G_cycle + 5 * loss_G_idt
        
        # Backpropagation
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
```
### Module 2: Domain-Adversarial Training (DANN)
**Goal**: Align source and target features via adversarial learning.
**Code Implementation:**
```mips
# Domain-Adversarial Neural Network (DANN)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove FC layer
    
    def forward(self, x):
        return self.resnet(x)

class TaskClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(2048, num_classes)  # ResNet50 features

class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Domain probability

# Gradient Reversal Layer
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output  # Invert gradients

# Training Loop
feature_extractor = FeatureExtractor()
task_classifier = TaskClassifier(num_classes=19)  # e.g., Cityscapes classes
domain_classifier = DomainClassifier()

optimizer = torch.optim.Adam([
    {'params': feature_extractor.parameters()},
    {'params': task_classifier.parameters()},
    {'params': domain_classifier.parameters()}
], lr=1e-4)

for synth_imgs, real_imgs in zip(synth_loader, real_loader):
    # Feature extraction
    synth_feat = feature_extractor(synth_imgs)
    real_feat = feature_extractor(real_imgs)
    
    # Task loss (source only)
    task_output = task_classifier(synth_feat)
    loss_task = nn.CrossEntropyLoss()(task_output, synth_labels)
    
    # Domain loss
    domain_synth = domain_classifier(GradientReversal.apply(synth_feat))
    domain_real = domain_classifier(GradientReversal.apply(real_feat))
    loss_domain = nn.BCELoss()(domain_synth, torch.zeros_like(domain_synth)) + \
                  nn.BCELoss()(domain_real, torch.ones_like(domain_real))
    
    # Total loss
    total_loss = loss_task + 0.1 * loss_domain  # Adjust weight
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```
## Step 4： Self-Training Optimization
**Goal**: Refine the model using pseudo-labels from target domain predictions.
**Steps**:
1. **Initial Training**: Train on synthetic data.
2. **Pseudo-Label Generation**:
   ```mips
   model.eval()
   pseudo_labels = []
   conf_threshold = 0.9  # Confidence threshold

   with torch.no_grad():
       for real_imgs in real_loader:
           outputs = model(real_imgs)
           probs = torch.softmax(outputs, dim=1)
           max_probs, labels = torch.max(probs, dim=1)
           mask = (max_probs > conf_threshold)
           pseudo_labels.append(labels[mask])
   ```
3. **Mixed Training**:
   ```mips
   mixed_dataset = ConcatDataset([synth_dataset, pseudo_labeled_real_dataset])
   train_loader = DataLoader(mixed_dataset, batch_size=32, shuffle=True)
   ```
4. **Iterative Refinement**: Repeat steps 2-3 for multiple rounds.

## Step 5：Evaluation & Deployment
**Metrics**
- **Semantic Segmentation**: mIoU (mean Intersection-over-Union)
- **Object Detection**: mAP (mean Average Precision)
- **Domain Alignment**: FID (Fréchet Inception Distance)
  
**Deployment Optimization**
- **Model Compression**: Use knowledge distillation to shrink models (e.g., ResNet→MobileNetV3).
- **Hardware Acceleration**: Convert models to TensorRT for NVIDIA GPUs.
- **Edge Deployment**: Optimize for Jetson devices using ONNX Runtime or LibTorch.

## Troubleshooting
1. **Semantic Distortion in Translated Images**
   - **Solution**:Add a semantic consistency loss
   ```mips
   seg_model = pretrained_segmentation_model()
   seg_loss = nn.KLDivLoss()(seg_model(fake_B), seg_model(real_A))
   loss_G_total += 0.5 * seg_loss
   ```

2. **Poor Target Domain Performance**
- **Solution**: Use **Curriculum Learning**:
  - Start with easy real-world samples (e.g., daylight scenes).
  - Gradually introduce harder samples (e.g., night/low-light).

3. **High Computational Cost**
 - **Solution**: Enable mixed-precision training:
   ```mips
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()

   with autocast():
     outputs = model(inputs)
     loss = criterion(outputs, labels)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()

# Summary
This pipeline enables robust adaptation from synthetic to real domains by combining image translation, adversarial training, and self-training. Key considerations include balancing domain alignment with task performance and optimizing for deployment. Iterative refinement and careful hyperparameter tuning (e.g., loss weights, confidence thresholds) are critical for success.