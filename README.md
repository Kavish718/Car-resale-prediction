# Car Resale Price Prediction — Singapore

A machine learning project exploring used car resale price prediction in Singapore through feature engineering, regression modeling, neural networks, ensemble learning, and synthetic data augmentation.

The project investigates how different modeling approaches perform on tabular vehicle data, with experiments spanning traditional regression models, gradient boosting frameworks, neural network architectures, and generative models for synthetic data creation.

---

## Overview

Predicting used car resale prices is a challenging regression problem due to the interaction between vehicle specifications, depreciation, market demand, and categorical attributes.

This project explores multiple machine learning approaches to estimate resale prices, with an emphasis on model comparison, hyperparameter optimization, feature engineering, and experimentation with synthetic data augmentation.

Rather than relying on a single predictive model, the project evaluates different families of algorithms and investigates whether alternative architectures, ensemble strategies, and synthetic training data can improve prediction performance.

## Objectives

- Develop machine learning models for used car resale price prediction.
- Explore feature preprocessing, feature engineering, and dimensionality reduction.
- Compare traditional regression, gradient boosting, and neural network approaches.
- Optimize model performance through hyperparameter tuning.
- Experiment with synthetic tabular data generation using CVAE and CTGAN.
- Investigate ensemble learning through combining gradient boosting predictions.
- Analyze prediction errors and model behavior.

---

## Machine Learning Approaches

### 1. Traditional Regression Models

Baseline and conventional regression models were explored to establish performance comparisons across different modeling approaches.

- Linear Regression
- Ridge Regression
- Lasso Regression
- K-Nearest Neighbors (KNN)
- Random Forest Regressor

Hyperparameter optimization was performed for selected models using Grid Search and cross-validation.

### 2. Gradient Boosting Models

Tree-based boosting algorithms were explored to capture nonlinear relationships and complex interactions between vehicle features.

| Model | Framework | Approach |
|---|---|---|
| XGBoost | XGBoost | Gradient-boosted decision trees |
| LightGBM | LightGBM | Efficient gradient boosting |
| CatBoost | CatBoost | Gradient boosting with categorical-data support |

Experiments include model configuration, validation, and hyperparameter optimization for boosting-based regressors.

### 3. Neural Network Models

Custom neural network architectures were implemented using PyTorch to investigate nonlinear feature representations for price prediction.

The experiments include:

- Feedforward neural networks
- Residual neural network architectures
- Gated Linear Unit (GLU) networks

Training incorporates techniques such as:

- Feature and target normalization
- Adam optimization
- Regularization through weight decay
- Validation-based monitoring
- Early stopping
- Hyperparameter tuning
- Training and validation loss visualization

---

## Feature Engineering & Preprocessing

Data preprocessing and feature engineering form a significant part of the project pipeline.

The preprocessing workflow includes:

- Data cleaning and transformation
- Categorical feature encoding
- Numerical feature preprocessing
- Feature selection and transformation
- Feature scaling for models requiring normalized inputs
- Dimensionality reduction using Principal Component Analysis (PCA)

PCA was explored to transform selected groups of features into lower-dimensional representations, allowing experimentation with reduced feature spaces.

---

## Synthetic Data Generation

An additional experiment investigates whether synthetic tabular data can be used to augment the training dataset and influence downstream predictive performance.

Two generative approaches were explored:

### Conditional Variational Autoencoder (CVAE)

A conditional VAE was implemented using PyTorch to learn a latent representation of the feature-price distribution and generate synthetic samples.

The model uses a latent space representation to reconstruct features conditioned on the target price, allowing synthetic feature samples to be generated alongside price values.

### CTGAN

**Conditional Tabular GAN (CTGAN)** was used to generate synthetic tabular samples while accounting for categorical features.

The generated data was combined with the original dataset to create augmented datasets for downstream model training experiments.

### Data Augmentation Pipeline

```text
Original Training Data
        │
        ├── Conditional VAE
        │       └── Synthetic Samples
        │
        └── CTGAN
                └── Synthetic Samples
                        │
                        ▼
               Augmented Datasets
                        │
                        ▼
           Neural Network Training
                        │
                        ▼
             Validation & Comparison
```

The synthetic data experiments were designed to investigate the impact of generated samples on model performance rather than assume that augmentation necessarily improves prediction accuracy.

---

## Ensemble Learning

An ensemble experiment combines predictions from three gradient boosting models:

- XGBoost
- LightGBM
- CatBoost

The ensemble uses a weighted averaging strategy to combine individual model predictions.

\[
\hat{y}_{ensemble}
=
0.33\hat{y}_{XGB}
+
0.33\hat{y}_{LGBM}
+
0.33\hat{y}_{CatBoost}
\]

This approach explores whether combining different boosting implementations can produce more robust predictions than relying on a single model.

---

## Model Evaluation

Model performance is evaluated using regression metrics, including:

- **Root Mean Squared Error (RMSE)** — Measures prediction error while penalizing larger deviations.
- **Mean Squared Error (MSE)** — Measures the average squared difference between predicted and actual prices.
- **R² Score** — Measures the proportion of variance explained by the predictive model.
- **Mean Absolute Error (MAE)** — Measures the average absolute prediction error.

Validation experiments include prediction-error analysis, residual visualization, and comparison between individual models and ensemble predictions.

---

## Explainability

The neural network experiments incorporate **SHAP (SHapley Additive exPlanations)** using KernelExplainer to explore feature contributions to model predictions.

Feature importance visualizations are used to investigate how input features influence predicted resale prices and to improve interpretability of the modeling process.

---

## Project Structure

```text
Car-resale-prediction/
│
├── Source Code/
│   │
│   ├── Data_Preprocessing.ipynb
│   ├── Synthetic_Data_Generation.ipynb
│   ├── LightGBM_Training_and_Optimization.ipynb
│   ├── CatBoost_Training_and_Optimization.ipynb
│   ├── Non_NN_Models_with_Best_Params.ipynb
│   ├── NN_Hyperparameter_Tuning.ipynb
│   └── NN_Models.ipynb
│
└── README.md
```

### Notebook Descriptions

| Notebook | Description |
|---|---|
| `Data_Preprocessing.ipynb` | Exploratory data analysis and feature processing to generate training and testing datasets |
| `Synthetic_Data_Generation.ipynb` | Experiments with CVAE and CTGAN-based synthetic data generation and augmentation |
| `LightGBM_Training_and_Optimization.ipynb` | Training and hyperparameter tuning of LightGBM |
| `CatBoost_Training_and_Optimization.ipynb` | Training and hyperparameter tuning of CatBoost |
| `Non_NN_Models_with_Best_Params.ipynb` | Comparison and validation of non-neural-network models using selected hyperparameters |
| `NN_Hyperparameter_Tuning.ipynb` | Hyperparameter tuning for neural network variations |
| `NN_Models.ipynb` | Training and evaluation of neural network architectures |

---

## Tech Stack

### Languages & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch

### Machine Learning

- Regression
- Random Forest
- Ridge & Lasso
- K-Nearest Neighbors
- XGBoost
- LightGBM
- CatBoost
- Neural Networks
- Residual Networks
- Gated Linear Units (GLU)
- Ensemble Learning

### Generative Modeling

- Conditional Variational Autoencoders (CVAE)
- CTGAN
- Synthetic Tabular Data Generation

### Optimization & Analysis

- Grid Search
- Randomized Search
- Cross-Validation
- Hyperparameter Optimization
- PCA
- SHAP
- Exploratory Data Analysis
- Residual Analysis
