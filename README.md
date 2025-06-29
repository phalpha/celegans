# A Machine Learning Approach to the Automatic Classification of C. elegans

## Abstract

This research presents an automated approach to classifying C. elegans roundworms using convolutional neural networks. Using a fully convolutional U-net with a ResNet encoder, we performed semantic image segmentation on microscopy images from the Broad Bioimage Benchmark Collection. By labeling pixels as "alive," "dead," or "background," we accomplished two key classification tasks: determining whether entire images contain roundworms treated with amoxicillin, and classifying individual roundworms as dead or alive. Results demonstrate 100% accuracy in image-level classification and 93% accuracy in individual organism classification.

## Introduction

*Caenorhabditis elegans* is a valuable model organism for testing anti-infection drug therapies. When studying ampicillin effectiveness against bacterial infections, treated roundworms exhibit a curved, smooth "alive" phenotype, while untreated ones show a straight, uneven "dead" phenotype.

This research automates two classification tasks: determining if images contain treated C. elegans, and classifying individual roundworms as alive or dead based on morphological appearance using semantic segmentation with a U-net architecture.

## Methodology

**Dataset:** 100 grayscale microscopy images from the Broad Bioimage Benchmark Collection (52 untreated, 48 treated), containing 10-20 roundworms each. Images were resized from 696×520 to 704×512 pixels for model compatibility.

**Data Preparation:** Images split 80/20 for training/testing. Over 1,500 individual C. elegans were manually labeled using curvature heuristics, with pixels classified as background (0), alive (1), or dead (2).

**Model:** U-net with ResNet encoder trained for 10 epochs on Stanford's Sherlock computing cluster. The semantic segmentation model outputs pixel-wise classifications for entire images.

**Evaluation:** Image-level classification based on pixel count ratios (more "alive" pixels = untreated, more "dead" pixels = treated). Individual organism classification determined by dominant pixel values in each roundworm region.

## Results

**Image-Level Classification:** Achieved 100% accuracy, correctly identifying all 20 test images as treated or untreated.

**Individual Organism Classification:** Achieved 93% accuracy (255/275 organisms correctly classified), with 94% sensitivity for alive classification and 92.4% sensitivity for dead classification.

## Discussion

The results provide strong preliminary evidence for automated C. elegans classification using deep learning. The perfect image-level classification and high individual organism accuracy demonstrate the viability of this approach for high-throughput drug screening applications.

**Limitations:** The small dataset (100 images) limits robustness assessment, and some hyperparameters require optimization. Manual labeling introduces potential subjective bias.

**Future Work:** Plans include parameter optimization, dataset expansion through GANs and data augmentation, and comparative analysis with existing methods.

## Significance

This work demonstrates that semantic segmentation can effectively automate C. elegans classification, offering potential for scalable drug screening applications while providing detailed pixel-level analysis of organism morphology.

---

*Research conducted or for a at Stanford CS 279 Project, with computational support from the Stanford Sherlock computing cluster.*
