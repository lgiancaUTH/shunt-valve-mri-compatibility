# Automatic Brain Shunt Valve Recognition. Feature Analysis

This repository contains the supporting code for the manuscript:


>L Giancardo, O Arevalo, A Tenreiro, R Riascos and E Bonfante. MRI Compatibility: Automatic Brain Shunt Valve Recognition using Feature Engineering and Deep Convolutional Networks. Scientific Reports volume 8, Article number: 16052 (2018).
https://www.nature.com/articles/s41598-018-34164-6



In this manuscript, we describe a X-rays-based implanted device identification system for MRI safety and devise a pilot study to test the feasibility of the automatic image recognition component. We evaluate machine learning-based methods to identify different cerebrospinal fluid shunt valves from 416 skull X-rays clinical images. Our best performing method identified the valve type correctly 96% [CI 94-98%] of the times (CI: confidence intervals, precision 0.96, recall 0.96, f1-score 0.96), tested using a stratified cross-validation approach to avoid chances of overfitting.

## Installation
This code has been tested with Python 3.6 and pip 

```
git clone https://github.com/lgiancaUTH/shunt-valve-mri-compatibility.git
cd shunt-valve-mri-compatibility
```
Optionally, create a (conda) virtual environment and activate it
```
conda create -n valve-env python=3.6 

source activate valve-env
```

and finally, install the required dependencies
```
pip install -r requirements.txt
```

## Run
Run analysis using features derived from convolutional neural networks  
```
python runExpCNN-paper.py
```
Run analysis using features computed with Histogram of Oriented Gradients (HOG)  
```
python runExpHOG-paper.py
```
Run analysis using features computed with Local Binary Patterns (LBP)  
```
python runExpLBP-paper.py
```
## Other information
This code is free to use for any non-commercial purposes, provided that the original publication is cited. 

We refer to the original publication for additional infomation and acknowledgements.
