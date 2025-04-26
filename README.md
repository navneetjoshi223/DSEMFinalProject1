# Paper 1 - Do Generated Data Always Help Contrastive Learning?

This project explores the use of generated (synthetic) images to improve contrastive learning for self-supervised representation learning. It is based on the paper “Do Generated Data Always Help Contrastive Learning?” presented at ICLR 2024.

## Project Members
	•	Reetika Bhanushali
	•	Navneet Joshi
	•	Siddi Kommuri

## Guide
	•	Prof. Akash Murthy

## Overview

Contrastive learning (CL) methods like SimCLR and MoCo benefit from strong augmentations and diverse training data. Recently, synthetic data generated via diffusion models has been proposed to further improve CL performance. However, this project investigates and confirms that naively adding generated data can sometimes hurt performance.

We implement and evaluate Adaptive Inflation (AdaInf) — a strategy that:
	•	Adjusts the real-to-synthetic data mixing ratio (favoring real data 10:1)
	•	Applies weaker augmentations to avoid harming learning from synthetic data.

## Goals
	•	Diagnose when generated data helps or hurts CL
	•	Implement and validate the AdaInf strategy
	•	Evaluate across datasets (CIFAR-10, CIFAR-100, Tiny ImageNet)
	•	Test across multiple CL methods (SimCLR, MoCo v2, BYOL, Barlow Twins)

## Key Findings
	•	AdaInf consistently improves performance compared to naive vanilla inflation or no inflation.
	•	Proper data balancing and weaker augmentations are crucial for fully utilizing synthetic data.
	•	AdaInf particularly shines in data-scarce settings, boosting performance without requiring labeled data.

## Datasets Used
	•	CIFAR-10
	•	CIFAR-100
	•	Tiny ImageNet (subset)

## Frameworks Implemented
	•	SimCLR
	•	MoCo v2
	•	BYOL
	•	Barlow Twins

## Conclusion

Generated data can significantly benefit contrastive learning — but only with careful adjustments. Adaptive Inflation (AdaInf) proves to be a simple, effective, and generalizable strategy for improving self-supervised representation learning.

# How to run

	1.	Clone the repository
```git clone https://github.com/your-username/contrastive-learning-generated-data.git
cd contrastive-learning-generated-data```

	2.	Set up a Python environment
```python3 -m venv venv
source venv/bin/activate```

	3.	Install required packages
pip install -r requirements.txt
(Requirements include torch, torchvision, sololearn, and basic Python utilities.)

	4.	Download datasets
You can download CIFAR-10, CIFAR-100 automatically using torchvision. Example in Python:
from torchvision.datasets import CIFAR10
CIFAR10(root=’./data’, download=True)

Or through a script:
python download_cifar.py –dataset cifar10
python download_cifar.py –dataset cifar100

	5.	Prepare synthetic data
If you have a pretrained diffusion model, generate synthetic images.
If synthetic images are already provided, place them inside a folder like data/generated/.

	6.	Train models with different setups

	To train SimCLR without inflation (baseline):
	python train.py –method simclr –dataset cifar10 –inflation none
	
	To train SimCLR with vanilla inflation:
	python train.py –method simclr –dataset cifar10 –inflation vanilla
	
	To train SimCLR with Adaptive Inflation (AdaInf):
	python train.py –method simclr –dataset cifar10 –inflation adainf

(You can replace –method with moco, byol, or barlow_twins to run different frameworks.)

	7.	Evaluate with linear probing
After pretraining is done, evaluate the feature quality using:
python eval_linear.py –dataset cifar10 –method simclr
