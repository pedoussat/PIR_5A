# 5A-Research_Project: Transilien Passenger Affluence Prediction
This project implements LSTM deep learning models to predict passenger traffic across the Transilien network.
https://challengedata.ens.fr/participants/challenges/149/

## Project Architecture

### Exploration Notebooks
* **Exploration.ipynb**: Data analysis, preprocessing steps (COVID-19 period removal), and station data quality assessment.
* **OptimizeParameters3.ipynb**: Hyperparameter tuning for the models defined in version 3.
* **SeePredictionResults.ipynb**: Tool for loading, visualizing, and comparing generated predictions using Matplotlib plots.

### Modules
* **modelsV1.py**: LSTM architectures using **chronological validation** (the last 20% of 2022 is used as a validation set). Predictions rely solely on exogenous variables (`job`, `ferie`, `vacances`).
* **modelsV2.py**: Same architecture as V1, but uses **randomly sampled validation** across the entire training period to improve robustness.
* **utils.py**: Core utility toolbox for raw data cleaning, sequence generation, visualization, and final submission formatting.

### Prediction Attempts
* **LSTMn°3.ipynb**: Generation of the first and second submission sets (before/after optimization). **Note:** Trained without internal validation data.
* **LSTMn°4.ipynb**: Generation of the third submission set using internal validation data for better convergence monitoring.

### Data Files
* **train_f_x.csv**: Input features for training.
* **y_train_sncf.csv**: Target values (passenger counts).
* **x_test.csv**: Input features for the 2023 prediction period.

### Submission History (Attempts)
| File | Description | Score |
| :--- | :--- | :--- |
| **y_test_LSTM_v3.1_sorted** | Initial model with randomly choosen hyperparameters. | **219.14** |
| **y_test_LSTM_v3.2_sorted** | Optimized model after hyperparameter tuning. | **209.61** |
