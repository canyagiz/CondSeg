---
# CondSeg-PyTorch: Ellipse Estimation via Conditioned Segmentation (Extended for RGB)

## Overview

This repository provides an unofficial, PyTorch-based implementation of the CondSeg architecture, originally proposed in the paper *"CondSeg: Ellipse Estimation of Pupil and Iris via Conditioned Segmentation"*.

CondSeg bypasses the labor-intensive process of explicitly annotating full elliptical parameters. Instead, it formulates pupil and iris parsing as a conditioned segmentation task, utilizing visible-only mask annotations to estimate the 5D elliptical parameters  through a differentiable `Ellp2Mask` module.

### Architectural Enhancements & Custom Pipeline

While the original paper focused on infrared (IR) images from AR/VR headsets like the OpenEDS datasets, this implementation has been substantially extended and optimized for broader, high-precision industrial and medical applications:

* **Native RGB Compatibility:** The network backbone and feature extraction pipelines have been modified to process standard RGB color space inputs, extending the model's utility beyond traditional IR-illuminated environments.
* **Sub-Pixel Limbus Precision:** The pipeline is optimized for micro-movement detection, achieving a strictly validated **0.05px precision** on the limbus radius. This level of sub-pixel accuracy makes it highly suitable for demanding video processing tasks, such as measuring Ocular Pulse Amplitude (OPA) and tracking minute corneal deformations.
* **Extreme Data Efficiency:** Achieving the aforementioned 0.05px precision was accomplished using a highly constrained dataset of only **270 ground-truth samples**. This was made possible by coupling the CondSeg architecture with a heavy, domain-specific data augmentation and synthetic generation pipeline.

## Reference Paper

**Title:** CondSeg: Ellipse Estimation of Pupil and Iris via Conditioned Segmentation **Authors:** Zhuang Jia, Jiangfan Deng, Liying Chi, Xiang Long, and Daniel K. Du **Affiliation:** Bytedance Inc. **Link:** [arXiv:2408.17231v1](https://arxiv.org/abs/2408.17231) 

> 
> **Abstract Snippet:** Parsing of eye components (i.e. pupil, iris and sclera) is fundamental for eye tracking and gaze estimation... we design a novel method CondSeg to estimate elliptical parameters of pupil/iris directly from segmentation labels, without explicitly annotating full ellipses, and use eye-region mask to control the visibility of estimated pupil/iris ellipses. 
> 
> 

## Methodology & Architecture

The architecture relies on decoupling the prediction of the eye-region mask from the pupil/iris ellipses.

1. 
**Eye-Region Segmenter:** A dense-block based encoder-decoder extracts image features and predicts the pixel-wise eye-region mask (sclera + visible iris + visible pupil).


2. 
**Iris & Pupil Estimators:** The encoded features are passed through multi-layer perceptrons (MLPs) to predict normalized 5D elliptical parameters. The full-pupil estimation is bounded by cropping the full-iris Region of Interest (RoI) to force the network to focus on the relative location and scale.


3. **Differentiable Mask Conversion:** To train without 5D ground-truth labels, the predicted parameters are converted into a general ellipse matrix format:


 This creates a distance map that is passed through a Sigmoid function to generate a soft segmentation mask.


4. **Conditioned Assembly:** The predicted full-ellipse soft masks are multiplied by the predicted eye-region mask. The resulting visible-region masks are then optimized against standard segmentation ground-truths using a binary cross-entropy loss.



## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CondSeg-PyTorch.git
cd CondSeg-PyTorch

# Install dependencies
pip install -r requirements.txt

```

*Note: Ensure your environment supports PyTorch with CUDA acceleration for optimal training performance, particularly if adapting this to GCP or Vertex AI instances.*

## Quick Start (Inference)

```python
import torch
import cv2
from models.condseg import CondSegRGB

# Initialize the extended RGB model
model = CondSegRGB(input_channels=3, out_channels=3)
model.load_state_dict(torch.load('weights/condseg_rgb_0.05px.pth'))
model.eval()

# Load an RGB eye image
image = cv2.imread('sample_rgb_eye.jpg')
image_tensor = preprocess_image(image)

# Forward pass
with torch.no_grad():
    eye_mask, iris_params, pupil_params = model(image_tensor)

print(f"Predicted Limbus (Iris) Radius (a, b): {iris_params['a']}, {iris_params['b']}")

```

## Training Data & Augmentation Note

The provided pre-trained weights were achieved using 270 manually annotated RGB frames. To reproduce the sub-pixel accuracy, your dataloader must implement the required aggressive augmentation strategies (spatial transformations, noise injection, and color jittering) to prevent overfitting on small datasets while preserving the geometric integrity of the elliptical labels.

## Legal & Intellectual Property Disclaimer

This repository contains an independent, clean-room implementation of the algorithm described in the referenced paper. The original paper and its underlying methods were developed by researchers affiliated with Bytedance Inc. This code is provided strictly for academic, educational, and research purposes. Users are responsible for verifying intellectual property rights and patent statuses before utilizing this architecture in commercial or production environments.

## Citation

If you utilize the CondSeg architecture in your research, please cite the original authors:

```bibtex
@article{jia2024condseg,
  title={CondSeg: Ellipse Estimation of Pupil and Iris via Conditioned Segmentation},
  author={Jia, Zhuang and Deng, Jiangfan and Chi, Liying and Long, Xiang and Du, Daniel K},
  journal={arXiv preprint arXiv:2408.17231},
  year={2024}
}

```

---
