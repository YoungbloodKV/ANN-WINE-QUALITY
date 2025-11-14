Wine Quality Classification using Artificial Neural Networks (ANN)
<p align="center"> <img src="banner_image.png" width="100%"> </p>
1. Project Overview

This project implements a lightweight Artificial Neural Network (ANN) to classify wine samples as Good or Bad based on their physicochemical properties.
The workflow includes data preprocessing, ANN model training, evaluation, and performance visualization.
The entire solution is implemented in a single Jupyter Notebook (ANN.ipynb) along with the dataset file (WineQT.csv).

2. Project Files
ANN.ipynb        - Complete ANN model and workflow  
WineQT.csv       - Dataset for training and evaluation  
banner_image.png - Repository banner  
README.md        - Project documentation  

3. Objective

The goal of this project is to build an ANN that predicts whether a wine sample is of good quality.
The dataset contains chemical attributes such as acidity, alcohol, sulphates, pH, and density.

The quality score is converted into a binary classification target:

Good Quality (1): quality score ≥ 6

Bad Quality (0): quality score < 6

This classification enables the neural network to learn quality patterns from input features.

4. Dataset Description

WineQT.csv includes the following variables:

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

Missing values, if present, are removed using simple row-wise cleaning.

5. Model Architecture

The ANN defined in ANN.ipynb uses the following structure:

Input Layer: 12 input features

Hidden Layer 1: 12 neurons, ReLU activation

Hidden Layer 2: 8 neurons, ReLU activation

Output Layer: 1 neuron, Sigmoid activation

Loss Function: Binary Crossentropy

Optimizer: Adam

Metric: Accuracy

Training Duration: 50 epochs, batch size 32, with validation split

This architecture is simple, efficient, and suitable for binary classification.

6. Steps Followed in the Notebook
6.1 Data Loading

Load dataset using pandas

Inspect first few rows

Understand column definitions

6.2 Data Cleaning

Identify missing values

Remove them using dropna()

6.3 Label Transformation

Convert the numeric quality column into a binary label:

quality_label = 1 if quality ≥ 6 else 0

6.4 Feature Scaling

Normalize features using StandardScaler

6.5 Train-Test Split

Split dataset into training and testing sets (80:20)

6.6 ANN Model Creation

Build a Sequential neural network

Use ReLU activation in hidden layers

Use Sigmoid activation for binary output

6.7 Model Training

Train for 50 epochs

Monitor validation accuracy and loss

6.8 Model Evaluation

Generate the following:

Confusion matrix

Accuracy score

Classification report

Training and validation curves

6.9 Forward & Backward Pass Demonstration

The notebook includes:

Manual forward propagation

Gradient computation via tf.GradientTape()

This section explains ANN fundamentals and gradient flow.

7. How to Run the Project
Install dependencies:
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn

Execute the notebook:

Open ANN.ipynb in:

Jupyter Notebook

VS Code

Google Colab

Run all cells in order.

All results and visualizations appear inline in the notebook.

8. Future Enhancements

Add regularization (Dropout, L2, BatchNorm)

Increase hidden layers

Tune hyperparameters

Convert to multi-class prediction

Deploy using Streamlit, Flask, or FastAPI

Compare performance with Random Forest, SVM, and XGBoost

9. Repository Banner
![Project Banner](banner_image.png)