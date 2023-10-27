# PyTorch based GAN model for MNIST Dataset

## Overview

Have you ever wondered how photo editing applications are able to convert night images to day images? Also, you must have come across some photos of people who do not exist in the world. One of the powerful tools to achieve this is Generative Adversarial Networks, also known as GAN. GANs are mainly used in image-to-image translation and to generate photorealistic images that even a human fails to identify as fake or true.

---

## Aim

In this project the PyTorch framework is used to build the GAN model on the MNIST dataset. Finally, we learn how to use the Generator model for generating new images of digits.


---

## Tech Stack

- **Language:** `Python`
- **Libraries:** `torch`, `torchvision`, `numpy`, `matplotlib`

---

## Approach

1. Introduction to Generative Adversarial Network
2. Introduction to Generator and Discriminator
3. Loss function in GAN
4. Building Model on PyTorch
5. Model training on Google Colab
6. Generating fake images with Generator

---

## Modular Code Overview

1. **data**: Contains data_utils.py file, which is used to download and transform the data. It will download and store the data in the respective folder.
2. **model**: Contains gan.py
3. **run.py**: Contains the main code where all functions are called.
4. **train.py**: Contains the code for model training.
5. **requirements.txt**: Lists all the required libraries with respective versions. Install the file using the command `pip install -r requirements.txt`. Note: Please use CUDA versions of torch if CUDA is available.

---

## Key Concepts Explored

1. Generative Adversarial Network
2. Generator
3. Discriminator
4. The loss function of the Generator.
5. The Loss function of the Discriminator.
6. Transform data in PyTorch
7. GAN model from scratch in PyTorch
8. Generate a fake image using the Generator

---


