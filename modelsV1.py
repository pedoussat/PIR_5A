"""
models.py
=========

This module contains the deep learning architectures used for predicting 
passenger affluence in the Transilien train network.

The models are configured to perform predictions exclusively based on exogenous 
variables (job, ferie, vacances), meaning that each prediction is independent of 
previously predicted target values. 

METHODOLOGY:
------------
1. Local Modeling: Each station is treated independently with its own LSTM model.
2. Time Series Sequencing: We use a random sliding window approach (seq_len) to capture 
   temporal dependencies from exogenous features (calendars).
3. Chronological Validation: Training is split 80/20 using a time-ordered cut 
   to prevent data leakage and evaluate performance on a recent 'validation' period.
4. Regularization: Implementation of EarlyStopping to prevent overfitting 
   by monitoring 'val_loss'.

MAIN FUNCTIONS:
---------------
- lstm_model: Comprehensive training/validation loop for model exploration 
  and hyperparameter tuning (includes MAPE results). 

- submission_prediction_QD6: Targeted prediction function for station 'QD6'.
- lstm_submission_prediction_all: Final production loop for generating 
  2023 predictions across all 438 stations.

DEPENDENCIES:
-------------
- TensorFlow/Keras: Sequential API with LSTM layers.
- Scikit-learn: MinMaxScaler for feature normalization.
- Utils: Custom functions for sequence generation.
"""


# IMPORTS
# ===============================================================================================================================================================================================
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


# MODELS FOR PREDICTION OVER 2022 (allow for mape_results[])
# ===============================================================================================================================================================================================
# ===============================================================================================================================================================================================

# LSTM MODEL EXPLORATION RUN FOR ALL STATIONS (prediction over 2022)
# ===============================================================================================================================================================================================
# Call
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# sample_test, mape_results, all_losses = models.lstm_model(sample_train, sample_test, sample_test_true, sample_size, seq_len, units, activation, learning_rate, batch_size, epochs, early_stop = False, features = ['job', 'ferie', 'vacances'])

# Code
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def lstm_model(sample_train,
               sample_test,
               sample_test_true,
               sample_size,
               seq_len, units, activation, learning_rate, batch_size,
               epochs,
               early_stop = False,
               features = ['job', 'ferie', 'vacances']):
    idx=1
    mape_results = []
    all_losses = {}

    for name_station in sample_train.keys():
        try:
            # START THE LOOP
            # ===================================
            print(f"{idx}/{sample_size} Station {name_station}")
            idx+=1
            
            # DATA EXTRACTION
            # ===================================
            df_train = sample_train[name_station]    # 2015-01 → 2022-05
            df_test = sample_test[name_station]      # 2022-06 → 2022-12

            # FEATURES
            # ===================================
            X_train = df_train[features]
            y_train = df_train['y']
            X_test = df_test[features]

            # SCALING
            # ===================================
            scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
            X_test_scaled = scaler_X.transform(X_test)

            # SEQUENCES TRAIN
            # ===================================
            X_train_seq, y_train_seq = utils.create_sequences_random(
                pd.DataFrame(X_train_scaled),
                pd.DataFrame(y_train_scaled),
                seq_len
            )

            # CHRONOLOGICAL SPLIT BETWEEN LEARNING AND VALIDATION DATA
            # ========================================================
            split = int(len(X_train_seq) * 0.8)

            X_train_final = X_train_seq[:split]
            y_train_final = y_train_seq[:split]

            X_val = X_train_seq[split:]
            y_val = y_train_seq[split:]

            # SEQUENCES TEST
            # ========================================================
            X_test_full = np.vstack([
                X_train_scaled[-seq_len:],  # fin 2022
                X_test_scaled               # début 2023
            ])

            X_test_seq = np.array([
                X_test_full[i:i+seq_len]
                for i in range(len(X_test))
            ])

            # MODEL CONSTRUCTION
            # ========================================================
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
            # ========================================================
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
            # ========================================================
            all_losses[name_station] = {
                'train': history.history['loss'],
                'val': history.history['val_loss']
            }
            # PREDICTION 2023
            # ========================================================
            y_pred_scaled = model.predict(X_test_seq)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            # MAPE SCORE
            # ========================================================
            df_test_true = sample_test_true[name_station] 
            score = mean_absolute_percentage_error(df_test_true['y'], y_pred)
            mape_results.append({"station": name_station,"MAPE": score})

            # ASSIGNMENT
            # ========================================================
            sample_test[name_station]['y'] = y_pred.flatten()

            # MEMORY CLEANUP (for large loops)
            # ========================================================
            K.clear_session()
            del model # delete the model object

        except Exception as e:
            print(f"Station {name_station} skipped due to error {e}")

    return sample_test, mape_results, all_losses







# MODELS FOR PREDICTION OVER 2023 (does'nt allow for mape_results[])
# ===============================================================================================================================================================================================
# ===============================================================================================================================================================================================

# RNN MODEL PREDICTION RUN FOR 'QD6' STATION (prediction over 2023)
# ===============================================================================================================================================================================================
# Call
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# y_predicted_vX, qd6_losses = models.submission_prediction_QD6(df_per_station_train, df_per_station_test, y_predicted_vX, seq_len, units, activation, learning_rate, batch_size, epochs, early_stop = False, features = ['job', 'ferie', 'vacances'])

# Code
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def submission_prediction_QD6(df_per_station_train,
                                   df_per_station_test,
                                   y_predicted_vX,
                                   seq_len, units, activation, learning_rate, batch_size, 
                                   epochs,
                                   early_stop = False,
                                   features = ['job', 'ferie', 'vacances']):
    qd6_losses = {}
    # START THE LOOP
    # ===================================
    name_station = 'QD6'
    print(f"Station {name_station}")

    # DATA EXTRACTION
    # =========================
    df_train = df_per_station_train[name_station]    # [2015-01 → 2022-12]
    df_test = df_per_station_test[name_station]      # [2023-01 → 2023-06]

    # FEATURES
    # =========================
    X_train = df_train[features]
    y_train = df_train['y']
    X_test = df_test[features]

    # Scaling
    # =========================
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)

    # SEQUENCES TRAIN
    # =========================
    X_train_seq, y_train_seq = utils.create_sequences_random(
        pd.DataFrame(X_train_scaled),
        pd.DataFrame(y_train_scaled),
        seq_len
    )

    # CHRONOLOGICAL SPLIT BETWEEN LEARNING AND VALIDATION DATA
    # ========================================================
    split = int(len(X_train_seq) * 0.8)

    X_train_final = X_train_seq[:split]
    y_train_final = y_train_seq[:split]

    X_val = X_train_seq[split:]
    y_val = y_train_seq[split:]

    # SEQUENCES TEST
    # =========================
    X_test_full = np.vstack([
        X_train_scaled[-seq_len:],  # end of 2022
        X_test_scaled               # beginning of 2023
    ])

    X_test_seq = np.array([
        X_test_full[i:i+seq_len]
        for i in range(len(X_test))
    ])

    # MODEL CONSTRUCTION
    # ========================================================
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
    # ========================================================
    fit_args = {
        "x": X_train_final, # On utilise uniquement la partie Train Final
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
    # ========================================================
    qd6_losses[name_station] = {
        'train': history.history['loss'],
        'val': history.history['val_loss']
    }

    # PREDICTION 2023
    # =========================
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # ASSIGNMENT
    # =========================
    y_predicted_vX.loc[y_predicted_vX['index'].str.contains(name_station), 'y'] = y_pred

    # MEMORY CLEANUP (for large loops)
    # ========================================================
    K.clear_session()
    del model # delete the model object

    return y_predicted_vX, qd6_losses



# ===============================================================================================================================================================================================
# LSTM MODEL PREDICTION RUN FOR ALL STATION (prediction over 2023)
# ===============================================================================================================================================================================================
# Call
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# y_predicted_vX, all_losses = models.lstm_submission_prediction_all(df_per_station_train, df_per_station_test, y_predicted_vX, seq_len, units, activation, learning_rate, batch_size, epochs, early_stop = False, features = ['job', 'ferie', 'vacances'])

# Code
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def lstm_submission_prediction_all(df_per_station_train,
                                   df_per_station_test,
                                   y_predicted_vX,
                                   seq_len, units, activation, learning_rate, batch_size, 
                                   epochs,
                                   early_stop = False,
                                   features = ['job', 'ferie', 'vacances']):
    idx=1
    total_stations = len(df_per_station_test.keys())
    all_losses = {}

    for name_station in df_per_station_test.keys():
        try:
            # START THE LOOP
            # ===================================
            print(f"{idx}/{total_stations} Station {name_station}")
            idx+=1

            # DATA EXTRACTION
            # =========================
            df_train = df_per_station_train[name_station]    # [2015-01 → 2022-12]
            df_test = df_per_station_test[name_station]      # [2023-01 → 2023-06]

            # FEATURES
            # =========================
            X_train = df_train[features]
            y_train = df_train['y']
            X_test = df_test[features]

            # Scaling
            # =========================
            scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
            X_test_scaled = scaler_X.transform(X_test)

            # SEQUENCES TRAIN
            # =========================
            X_train_seq, y_train_seq = utils.create_sequences_random(
                pd.DataFrame(X_train_scaled),
                pd.DataFrame(y_train_scaled),
                seq_len
            )

            # CHRONOLOGICAL SPLIT BETWEEN LEARNING AND VALIDATION DATA
            # ========================================================
            split = int(len(X_train_seq) * 0.8)

            X_train_final = X_train_seq[:split]
            y_train_final = y_train_seq[:split]

            X_val = X_train_seq[split:]
            y_val = y_train_seq[split:]

            # SEQUENCES TEST
            # =========================
            X_test_full = np.vstack([
                X_train_scaled[-seq_len:],  # end of 2022
                X_test_scaled               # beginning of 2023
            ])

            X_test_seq = np.array([
                X_test_full[i:i+seq_len]
                for i in range(len(X_test))
            ])

            # MODEL CONSTRUCTION
            # ========================================================
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
            # ========================================================
            fit_args = {
                "x": X_train_final, # On utilise uniquement la partie Train Final
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
            # ========================================================
            all_losses[name_station] = {
                'train': history.history['loss'],
                'val': history.history['val_loss']
            }

            # PREDICTION 2023
            # =========================
            y_pred_scaled = model.predict(X_test_seq)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            # ASSIGNMENT
            # =========================
            y_predicted_vX.loc[y_predicted_vX['index'].str.contains(name_station), 'y'] = y_pred

            # MEMORY CLEANUP (for large loops)
            # ========================================================
            K.clear_session()
            del model # delete the model object

        except Exception as e:
            print(f"Station {name_station} skipped due to error {e}")

    return y_predicted_vX, all_losses
