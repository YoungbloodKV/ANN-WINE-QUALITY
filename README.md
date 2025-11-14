Wine Quality Classification using Artificial Neural Networks (ANN)
<p align="center"> <img src="banner_image.png" width="100%"> </p>
1. Project Overview

This project implements a lightweight Artificial Neural Network (ANN) to classify wine samples as Good or Bad based on their chemical composition.
The workflow includes data preprocessing, ANN model training, evaluation, and performance visualization.
All logic is contained within a single Jupyter Notebook (ANN.ipynb) supported by the dataset (WineQT.csv).

2. Project Files
ANN.ipynb        - Complete ANN model and workflow
WineQT.csv       - Dataset used for model training and testing
banner_image.png - Repository banner
README.md        - Documentation

3. Objective

The goal is to predict wine quality using ANN.
The dataset provides physicochemical parameters such as acidity, pH, alcohol, sulphates, and density.

The quality score is converted into a binary label:

Good Quality (1) → quality ≥ 6

Bad Quality (0) → quality < 6

The ANN learns patterns between the features and the quality label.

4. Dataset Description

WineQT.csv contains the following attributes:

fixed acidity

volatile acidity

citric acid

residual sugar

chlorides

free sulfur dioxide

total sulfur dioxide

density

pH

sulphates

alcohol

quality

Basic cleaning is applied using dropna() if necessary.

5. Model Architecture

The ANN architecture defined in the notebook is:

Input Layer: 12 features

Hidden Layer 1: 12 neurons, ReLU

Hidden Layer 2: 8 neurons, ReLU

Output Layer: 1 neuron, Sigmoid

Loss Function: Binary Crossentropy

Optimizer: Adam

Metric: Accuracy

Training: 50 epochs, batch size 32, with validation split

This setup provides a simple but effective classification model.

6. Steps Followed
6.1 Data Loading

Importing and inspecting the dataset using Pandas.

6.2 Data Cleaning

Removing missing values using dropna().

6.3 Label Transformation

Creating a new binary column:

quality_label = 1 if quality ≥ 6 else 0

6.4 Feature Scaling

Using StandardScaler to normalize input variables.

6.5 Train-Test Split

Splitting into 80% training and 20% testing.

6.6 Model Creation

Building a Sequential ANN model with ReLU and Sigmoid layers.

6.7 Model Training

Training for 50 epochs with validation monitoring.

6.8 Model Evaluation

Generating:

Accuracy

Confusion matrix

Classification report

Training and validation curves

6.9 Forward & Backward Pass Illustration

The notebook includes:

Manual forward propagation

Gradient calculation with tf.GradientTape()

This section demonstrates ANN fundamentals in detail.

7. How to Run This Project
Install Dependencies
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn

Execute the Notebook

Open ANN.ipynb and run all cells sequentially in Jupyter Notebook or VS Code.

All results, graphs, and metrics will appear inline.

8. Future Enhancements

Additional hidden layers

Regularization (Dropout, BatchNorm)

Hyperparameter optimization

Multi-class quality prediction

Deployment using Streamlit, Flask, or FastAPI

Comparison with Random Forest, SVM, XGBoost

9. Repository Banner
![Project Banner](banner_image.png)