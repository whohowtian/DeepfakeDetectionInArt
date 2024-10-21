# IS424G2T32024

# Deepfake Detection Using CNN, SVM, and GAN

## Project Overview
This project aims to develop a robust system for detecting deepfake images of the inpainted class by combining three powerful machine learning models: Convolutional Neural Network (CNN), Support Vector Machine (SVM), and Generative Adversarial Network (GAN). The goal is to create a multi-model approach that leverages the strengths of each technique to improve the accuracy and reliability of deepfake detection.

### Why Deepfake Detection?
With the rapid advancement of AI, deepfake technology has made it possible to generate realistic fake images and videos, which can be used for malicious purposes such as misinformation, identity theft, and more. This project seeks to address these challenges by developing a solution capable of identifying altered images and preventing the spread of misinformation for inpainted artworks.

## Project Architecture
The solution consists of three main components:

1. **Convolutional Neural Network (CNN)**:
   - A CNN is used for feature extraction from input images.
   - It captures spatial hierarchies in the image data, which helps in distinguishing real images from altered ones.
   - The extracted features are then used as input for further classification.

2. **Support Vector Machine (SVM)**:
   - The features extracted from the CNN are fed into an SVM for binary classification (real vs. fake).
   - SVM is known for its effectiveness in handling high-dimensional data, making it suitable for this use case.

3. **Generative Adversarial Network (GAN)**:
   - A GAN is employed to generate synthetic data for training and to simulate realistic altered images.
   - It helps in creating adversarial examples and improving the robustness of the CNN and SVM models.
   - The GAN-based component can also be used for anomaly detection by comparing the original and generated images.

## Workflow
1. **Data Collection**:
   - Dataset was taken from [this Google Drive link](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view?usp=drivesdk). If you use the refined Wikiart dataset, please cite the following paper:

@article{artgan2018,
  title={Improved ArtGAN for Conditional Synthesis of Natural Image and Artwork},
  author={Tan, Wei Ren and Chan, Chee Seng and Aguirre, Hernan and Tanaka, Kiyoshi},
  journal={IEEE Transactions on Image Processing},
  volume    = {28},
  number    = {1},
  pages     = {394--409},
  year      = {2019},
  url       = {https://doi.org/10.1109/TIP.2018.2866698},
  doi       = {10.1109/TIP.2018.2866698}
}

2. **Preprocessing**:
- Resize and normalize the images.
- Augment the dataset to enhance model generalization.

3. **Feature Extraction (CNN)**:
- Train a CNN model to extract features from the input images.
- Use pre-trained models (e.g., VGG16, ResNet) as a starting point for transfer learning.

4. **Classification (SVM)**:
- Train an SVM model on the features extracted by the CNN.
- Perform hyperparameter tuning to optimize the SVM for accuracy.

5. **Data Augmentation and Adversarial Training (GAN)**:
- Train a GAN to generate synthetic deepfake images.
- Use these synthetic images to improve the training of the CNN and SVM.

6. **Model Evaluation**:
- Evaluate the combined model's performance using metrics such as accuracy, precision, recall, and F1-score.
- Perform cross-validation to assess the robustness of the system.

## Installation
To run this project, you'll need the following dependencies:

- Python 3.10 or higher
- PyTorch (for CNN and GAN implementation)
- Scikit-learn (for SVM)
- OpenCV (for image processing)
- NumPy and Pandas (for data manipulation)
- Matplotlib (for plotting and visualization)
- Pillow (for image handling)
- diffusers (for Stable Diffusion inpainting pipeline)

You can install the required libraries using:
```bash
pip install torch scikit-learn opencv-python-headless numpy pandas matplotlib pillow diffusers
