# VFX2023 - Group26 Homework

## Overview
This repository contains two assignments for the **VFX2023** course, focusing on **High Dynamic Range Imaging (HDR)** and **Panorama Stitching**. Each assignment implements key image processing techniques to achieve high-quality image reconstruction and blending.

## Homework 1: High Dynamic Range Imaging (HDR)
### Goal
Reconstruct **HDR images** from multiple exposures and apply **tone mapping** to preserve details in both bright and dark regions.

### Methods
- **Image Alignment**: Uses **Median Threshold Bitmap** for reducing misalignment due to slight camera movements.
- **HDR Reconstruction**: Implements **Paul Debevec's HDR method** to compute radiance maps.
- **Tone Mapping**: Uses **Bilateral Filtering** to adjust contrast while preserving local details.

### Results
- Successfully generated HDR images with improved **contrast and detail visibility**.
- Demonstrated the impact of **alignment** on image sharpness.

### Tools & Libraries
- Python, OpenCV, NumPy

## Homework 2: Panorama Stitching
### Goal
Stitch multiple images together into a seamless **panoramic image** using feature-based methods.

### Methods
- **Feature Detection**: Uses **Harris Corner Detector** to find key points.
- **Feature Description**: Implements **SIFT descriptor** for robust matching.
- **Feature Matching**: Uses **L2 distance and RANSAC** to refine correspondences.
- **Image Blending**: Warps images and applies blending to remove visible seams.

### Results
- Successfully stitched images into a **seamless panorama**.
- Compared different transformations (**Homography vs. Scale + Translation**) based on camera motion.

### Tools & Libraries
- Python, OpenCV, NumPy, Matplotlib

## Authors
- **Wei Ting-Yu (魏廷宇) - R11942104**
- **Charles Huang (黃湛元) - R11942180**

## References
- Paul Debevec’s HDR Method (SIGGRAPH 1997)
- Recognising Panoramas by Brown and Lowe (ICCV 2003)

## License
This project is for educational purposes only.

