# Wine Quality Classification using Artificial Neural Networks (ANN)

```{=html}
<p align="center">
```
`<img src="banner_image.png" width="100%">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 1. Project Overview

This project implements a lightweight Artificial Neural Network (ANN) to
classify wine samples as Good or Bad based on their physicochemical
properties. The workflow includes data preprocessing, ANN model
training, evaluation, and performance visualization. The entire project
is implemented within a single Jupyter Notebook (`ANN.ipynb`) supported
by the dataset (`WineQT.csv`).

------------------------------------------------------------------------

## 2. Project Files

    ANN.ipynb        - Complete ANN workflow and model implementation
    WineQT.csv       - Dataset used for training and evaluation
    banner_image.png - Repository banner
    README.md        - Documentation

------------------------------------------------------------------------

## 3. Objective

The objective is to predict wine quality using an ANN model. The dataset
provides chemical attributes such as acidity, pH, alcohol content,
sulphates, and density.

The numerical `quality` column is converted into a binary target:

-   Good Quality (1): quality ≥ 6
-   Bad Quality (0): quality \< 6

This classification allows the ANN to learn relationships between input
features and wine quality.

------------------------------------------------------------------------

## 4. Dataset Description

The dataset includes the following features:

-   fixed acidity
-   volatile acidity
-   citric acid
-   residual sugar
-   chlorides
-   free sulfur dioxide
-   total sulfur dioxide
-   density
-   pH
-   sulphates
-   alcohol
-   quality

Simple cleaning is applied using `dropna()` if any missing values exist.

------------------------------------------------------------------------

## 5. Model Architecture

The ANN model defined in the notebook has the following structure:

-   Input Layer: 12 features
-   Hidden Layer 1: 12 neurons (ReLU)
-   Hidden Layer 2: 8 neurons (ReLU)
-   Output Layer: 1 neuron (Sigmoid)
-   Loss Function: Binary Crossentropy
-   Optimizer: Adam
-   Metric: Accuracy
-   Training Epochs: 50
-   Batch Size: 32
-   Validation Split: 20%

------------------------------------------------------------------------

## 6. Steps Followed in the Notebook

### 6.1 Data Loading

-   Load dataset using pandas
-   Display first few rows
-   Inspect structure and column types

### 6.2 Data Cleaning

-   Check for missing values
-   Remove missing rows using `dropna()`

### 6.3 Label Transformation

Convert the `quality` score into a binary label:

    quality_label = 1 if quality ≥ 6 else 0

### 6.4 Feature Scaling

Normalize input variables using `StandardScaler`.

### 6.5 Train-Test Split

Split the dataset into 80% training and 20% testing.

### 6.6 ANN Model Creation

Define a Sequential model using ReLU activations and a final Sigmoid
output neuron.

### 6.7 Model Training

Train the model for 50 epochs with validation monitoring.

### 6.8 Model Evaluation

Generate:

-   Accuracy score
-   Confusion matrix
-   Classification report
-   Accuracy and loss curves

### 6.9 Forward & Backward Pass Demonstration

Includes manual forward propagation and gradient computation using
`tf.GradientTape()`.

------------------------------------------------------------------------

## 7. How to Run This Project

### Install required packages:

    pip install numpy pandas scikit-learn tensorflow matplotlib seaborn

### Execute the notebook:

-   Open `ANN.ipynb` in Jupyter Notebook, VS Code, or Google Colab
-   Run all cells sequentially

------------------------------------------------------------------------

## 8. Future Enhancements

-   Add deeper layers and regularization
-   Tune hyperparameters
-   Extend to multi-class prediction
-   Deploy model using Streamlit or Flask
-   Compare with XGBoost, SVM, Random Forest

------------------------------------------------------------------------

## 9. Repository Banner Reference

    ![Project Banner](banner_image.png)
