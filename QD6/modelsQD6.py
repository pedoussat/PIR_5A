"""
modelsQD6.py
=========

This module contains the deep learning architectures used for predicting 
passenger affluence for station QD6
"""

# IMPORTS
# ===============================================================================================================================================================================================
import sys
import os

# Add parent directory to PYTHONPATH to allow shared imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import utils

import pandas as pd
import numpy as np

import sklearn
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, SimpleRNN
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error
from keras.callbacks import EarlyStopping

# BACKTEST (prediction over 2022)
# ===============================================================================================================================================================================================
# Call
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# y_predicted_qd6, qd6_losses = modelsQD6.backtest_prediction_QD6(df_qd6_train, df_qd6_test, y_predicted_qd6, seq_len, units, activation, learning_rate, batch_size, epochs, early_stop = True, features = ['job', 'ferie', 'vacances'])
#
# Code
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def backtest_prediction_QD6(df_per_station_train,
                                   df_per_station_test,
                                   y_predicted_vX,
                                   seq_len, units, activation, learning_rate, batch_size, 
                                   epochs,
                                   early_stop = True,
                                   features = ['job', 'ferie', 'vacances'], architecture = 'rnn'):
    
    # START THE LOOP
    qd6_losses = {}
    qd6_mape_results = []
    name_station = 'QD6'
    print(f"Station {name_station}")

    # DATA EXTRACTION
    df_train = df_per_station_train.copy()
    df_test = df_per_station_test.copy()

    # FEATURES
    X_train = df_train[features]
    y_train = df_train['y']
    X_test = df_test[features]

    # SCALING
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)

    # SEQUENCES TRAIN
    X_train_seq, y_train_seq = utils.create_sequences_random(
        pd.DataFrame(X_train_scaled),
        pd.DataFrame(y_train_scaled),
        seq_len
    )

    # CHRONOLOGICAL SPLIT BETWEEN LEARNING AND VALIDATION DATA
    split = int(len(X_train_seq) * 0.8)
    X_train_final = X_train_seq[:split]
    y_train_final = y_train_seq[:split]
    X_val = X_train_seq[split:]
    y_val = y_train_seq[split:]

    # SEQUENCES TEST
    X_test_full = np.vstack([
        X_train_scaled[-seq_len:],  # end of 2022
        X_test_scaled               # beginning of 2023
    ])
    X_test_seq = np.array([
        X_test_full[i:i+seq_len]
        for i in range(len(X_test))
    ])

    # MODEL CONSTRUCTION
    if (architecture == 'rnn'):
        model = Sequential([
                Input(shape=(seq_len, X_train.shape[1])),
                SimpleRNN(units=units, activation=activation),
                Dense(1)
            ])
    elif (architecture == 'lstm'):
        model = Sequential([
                Input(shape=(seq_len, X_train.shape[1])),
                LSTM(units=units, activation=activation),
                Dense(1)
            ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse'
    )

    # MODEL TRAINING
    fit_args = {
        "x": X_train_final,
        "y": y_train_final,
        "validation_data": (X_val, y_val),
        "batch_size": batch_size,
        "verbose": 0
    }
    if early_stop:
        # EarlyStopping config:
        # - monitor: tracking validation loss to detect overfitting
        # - patience: allow 3 epochs without improvement before stopping
        # - restore_best_weights: revert to the best model version, not the last one
        callback = EarlyStopping(
            monitor='val_loss', 
            min_delta=0.0001,   # Minimum change to qualify as an improvement
            patience=3, 
            restore_best_weights=True
        )

        history = model.fit(**fit_args, epochs=100, callbacks=[callback])
    else:
        history = model.fit(**fit_args, epochs=epochs)

    # SAVE TRAINING AND VALIDATION LOSSES
    qd6_losses[name_station] = {
        'train': history.history['loss'],
        'val': history.history['val_loss']
    }

    # PREDICTION
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # ASSIGNMENT
    y_predicted_vX.loc[y_predicted_vX['index'].str.contains(name_station), 'y'] = y_pred

    # MAPE SCORE
    score = mean_absolute_percentage_error(df_test['y'], y_pred)
    qd6_mape_results.append({"station": name_station,"MAPE": score})

    # MEMORY CLEANUP
    K.clear_session()
    del model

    return y_predicted_vX, qd6_losses, qd6_mape_results


# SUBMISSION (prediction over 2023)
# ===============================================================================================================================================================================================
# Call
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# y_predicted_qd6, qd6_losses = modelsQD6.backtest_prediction_QD6(df_qd6_train, df_qd6_test, y_predicted_qd6, seq_len, units, activation, learning_rate, batch_size, epochs, early_stop = True, features = ['job', 'ferie', 'vacances'])
# 
# Code
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def submission_prediction_QD6(df_per_station_train,
                                   df_per_station_test,
                                   y_predicted_vX,
                                   seq_len, units, activation, learning_rate, batch_size, 
                                   epochs,
                                   early_stop = True,
                                   features = ['job', 'ferie', 'vacances'],
                                   architecture = 'rnn'):
    
    qd6_losses = {}

    # START THE LOOP
    name_station = 'QD6'
    print(f"Station {name_station}")

    # DATA EXTRACTION
    df_train = df_per_station_train.copy()
    df_test = df_per_station_test.copy()

    # FEATURES
    X_train = df_train[features]
    y_train = df_train['y']
    X_test = df_test[features]

    # Scaling
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)

    # SEQUENCES TRAIN
    X_train_seq, y_train_seq = utils.create_sequences_random(
        pd.DataFrame(X_train_scaled),
        pd.DataFrame(y_train_scaled),
        seq_len
    )

    # CHRONOLOGICAL SPLIT BETWEEN LEARNING AND VALIDATION DATA
    split = int(len(X_train_seq) * 0.9)

    X_train_final = X_train_seq[:split]
    y_train_final = y_train_seq[:split]

    X_val = X_train_seq[split:]
    y_val = y_train_seq[split:]

    # SEQUENCES TEST
    X_test_full = np.vstack([
        X_train_scaled[-seq_len:],  # end of 2022
        X_test_scaled               # beginning of 2023
    ])

    X_test_seq = np.array([
        X_test_full[i:i+seq_len]
        for i in range(len(X_test))
    ])

    # MODEL CONSTRUCTION
    if (architecture == 'rnn'):
        model = Sequential([
                Input(shape=(seq_len, X_train.shape[1])),
                SimpleRNN(units=units, activation=activation),
                Dense(1)
            ])
    elif (architecture == 'lstm'):
        model = Sequential([
                Input(shape=(seq_len, X_train.shape[1])),
                LSTM(units=units, activation=activation),
                Dense(1)
            ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse'
    )

    # MODEL TRAINING
    fit_args = {
        "x": X_train_final,
        "y": y_train_final,
        "validation_data": (X_val, y_val),
        "batch_size": batch_size,
        "verbose": 0
    }
    if early_stop:
        # EarlyStopping config:
        # - monitor: tracking validation loss to detect overfitting
        # - patience: allow 3 epochs without improvement before stopping
        # - restore_best_weights: revert to the best model version, not the last one
        callback = EarlyStopping(
            monitor='val_loss', 
            min_delta=0.0001,   # Minimum change to qualify as an improvement
            patience=3, 
            restore_best_weights=True
        )

        history = model.fit(**fit_args, epochs=100, callbacks=[callback])
    else:
        history = model.fit(**fit_args, epochs=epochs)

    # SAVE TRAINING AND VALIDATION LOSSES
    qd6_losses[name_station] = {
        'train': history.history['loss'],
        'val': history.history['val_loss']
    }

    # PREDICTION 2023
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # ASSIGNMENT
    y_predicted_vX.loc[y_predicted_vX['index'].str.contains(name_station), 'y'] = y_pred

    # MEMORY CLEANUP
    K.clear_session()
    del model

    return y_predicted_vX, qd6_losses


# SHOW RESULT
# ===============================================================================================================================================================================================
import matplotlib.pyplot as plt

# Functions for showing results
def show_predictions_results_one_station(df_train, df_test, df_true, station_name, model_name):
    """
    Plot training, predicted and true values for one station.
    """
    # Figure
    plt.figure(figsize=(20,6))

    plt.plot(df_train['date'],  # TRAIN
             df_train['y'],
             label='Training Values',
             color='blue',
             alpha=0.5)
    
    plt.plot(df_test['date'],  # TEST
             df_test['y'],
             label='Predicted Values',
             color='green')

    plt.plot(df_true['date'],  # TRUE
             df_true['y'],
             label='Actual Values',
             color='black',
             alpha=0.5)

    # Titles
    plt.title(f'Actual vs Predicted Values for station {station_name} with {model_name}')
    plt.xlabel('Data Point Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def show_predictions_two_results_one_station(df_train, df_test1, df_test2, df_true, station_name, model_name1, model_name2):
    """
    Plot training, predicted and true values for one station with enlarged labels.
    """
    # Figure
    plt.figure(figsize=(20, 6))

    plt.plot(df_train['date'],  # TRAIN
             df_train['y'],
             label='Training Values',
             color='blue',
             alpha=0.5)
    
    plt.plot(df_test1['date'],  # TEST 1
             df_test1['y'],
             label=f'Predicted Values with {model_name1}',
             color='green')
    
    plt.plot(df_test2['date'],  # TEST 2
             df_test2['y'],
             label=f'Predicted Values with {model_name2}',
             color='red')

    plt.plot(df_true['date'],  # TRUE
             df_true['y'],
             label='Actual Values',
             color='black',
             alpha=0.5)

    # Customizing sizes
    plt.title(f'Actual vs Predicted Values for station {station_name}', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Passenger Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Enlarging the Legend
    # 'prop' controls the font properties inside the legend
    plt.legend(fontsize=14, loc='upper left', frameon=True)
    
    plt.grid(True, alpha=0.3) # Added grid for better readability
    plt.show()

def show_prediction(df_train, df_test, station_name, model_name, start_date='2015-01-01'):
    """
    Plot training and predicted values for all stations.
    """
    # Extracting Data
    df_train_station = df_train
    df_test_station = df_test

    # Display Window
    start = pd.to_datetime(start_date, format='%Y-%m-%d')
    df_train_station= df_train_station[df_train_station['date'] > start]

    # Figure
    plt.figure(figsize=(20,6))

    plt.plot(df_train_station['date'],
             df_train_station['y'],
             label='Actual Values',
             color='blue',
             marker='o',
             markersize=4)

    plt.plot(df_test_station['date'],
             df_test_station['y'],
             label='Predicted Values',
             color='green',
             marker='o',
             markersize=4)
    
    # Titles
    plt.title(f'Predicted Values for station {station_name} with {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()