# Multi-branch Network with Multi-feature Enhancement for Improving the Generalization of Facial Forgery Detection

## Introduction

This is the implementation of Multi-branch Network with Multi-feature Enhancement for Improving the Generalization of Facial Forgery Detection


## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
  - [Model Training](#model-training)
  - [Model Testing](#model-testing)


## Requirements
<pre>
    * Python 3.x
    * PyTorch
    * numpy
    * torch
    * torchvision
    * tqdm
    * decord
    * dlib
    * opencv
    * face_recognition
    * timm
</pre>


## Usage

1. Clone this repository:

```bash
git clone https://github.com/daisy-12138/M2EH
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model Training

To train the M2EH model, follow these steps:

1. Prepare the training data according to the following structure and place it in the `sample_train_data` folder:
<pre>
    train:
        - fake
        - real
    valid:
        - fake
        - real
    test:
        - fake
        - real
</pre>
 

2. Run the training script:

```bash
python train.py
```

The model weights and metrics are saved in the `weight` folder.


## Model Testing

To run the code, use the following command:

```bash
python test.py
```
