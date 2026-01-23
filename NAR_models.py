"""
NAR_models.py
=========

This module contains the deep learning architectures used for predicting 
passenger affluence in the Transilien train network.

The models are configured to perform predictions exclusively based on exogenous 
variables (job, ferie, vacances), meaning that each prediction is independent of 
previously predicted target values.

Non Autoregressive (NAR) models.
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


# ===============================================================================================================================================================================================
# BACKTEST (prediction over 2022)
# ===============================================================================================================================================================================================

# Call
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# sample_test, mape_results, all_losses = models.lstm_model(df_train, df_test, df_true, sample_size, seq_len, units, activation, learning_rate, batch_size, epochs, early_stop = True, features = ['job', 'ferie', 'vacances'])

# Code
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def backtest_lstm(df_train_per_station, df_test_per_station, df_true_per_station, sample_size,
               seq_len, units, activation, learning_rate, batch_size, epochs,
               keep_percentage = 0.25,
               early_stop = True, features = ['job', 'ferie', 'vacances']):
    idx=1
    mape_results = []
    all_losses = {}

    for name_station in df_train_per_station.keys():
        try:
            # START THE LOOP
            # ===================================
            print(f"{idx}/{sample_size} Station {name_station}")
            idx+=1
            
            # DATA EXTRACTION
            # ===================================
            df_train = df_train_per_station[name_station].copy()    
            df_test = df_test_per_station[name_station].copy()      
            df_test_true = df_true_per_station[name_station].copy()

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

            # CHRONOLOGICAL SPLIT BETWEEN LEARNING AND VALIDATION DATA
            # ========================================================
            split = int(len(X_train_scaled) * 0.8)

            X_learn = X_train_scaled[:split]
            y_learn = y_train_scaled[:split]

            X_val = X_train_scaled[split:]
            y_val = y_train_scaled[split:]

            # LEARNING SEQUENCES
            # ===================================
            X_learn_seq, y_learn_seq = utils.create_sequences_uniform(
                pd.DataFrame(X_learn),
                pd.DataFrame(y_learn),
                seq_len,
                keep_percentage
            )

            # VALIDATION SEQUENCES
            # ===================================
            X_val_seq, y_val_seq = utils.create_sequences_uniform(
                pd.DataFrame(X_val),
                pd.DataFrame(y_val),
                seq_len,
                keep_percentage
            )

            # SEQUENCES TEST
            # ========================================================
            X_test_full = np.vstack([
                X_train_scaled[-seq_len:],  # end of 2022
                X_test_scaled               # begining of 2023
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
                "x": X_learn_seq,
                "y": y_learn_seq,
                "validation_data": (X_val_seq, y_val_seq),
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
            score = mean_absolute_percentage_error(df_test_true['y'], y_pred)
            mape_results.append({"station": name_station,"MAPE": score})

            # ASSIGNMENT
            # ========================================================
            df_test_per_station[name_station]['y'] = y_pred.flatten()

            # MEMORY CLEANUP (for large loops)
            # ========================================================
            K.clear_session()
            del model # delete the model object

        except Exception as e:
            print(f"Station {name_station} skipped due to error {e}")

    return df_test_per_station, mape_results, all_losses





# ===============================================================================================================================================================================================
# SUBMISSION (prediction over 2023)
# ===============================================================================================================================================================================================

# Call
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# df_predicted, all_losses = models.lstm_submission_prediction_all(df_train, df_test, df_predicted, seq_len, units, activation, learning_rate, batch_size, epochs, early_stop = True, features = ['job', 'ferie', 'vacances'])

# Code
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def submission_lstm(df_per_station_train,
                                   df_per_station_test,
                                   df_predicted,
                                   seq_len, units, activation, learning_rate, batch_size, 
                                   epochs,
                                   keep_percentage = 0.25,
                                   early_stop = True,
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
            df_train = df_per_station_train[name_station].copy()   
            df_test = df_per_station_test[name_station].copy()  
            df_predicted = df_predicted.copy()

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

            # CHRONOLOGICAL SPLIT BETWEEN LEARNING AND VALIDATION DATA
            # ========================================================
            split = int(len(X_train_scaled) * 0.8)

            X_learn = X_train_scaled[:split]
            y_learn = y_train_scaled[:split]

            X_val = X_train_scaled[split:]
            y_val = y_train_scaled[split:]

            # LEARNING SEQUENCES
            # ===================================
            X_learn_seq, y_learn_seq = utils.create_sequences_uniform(
                pd.DataFrame(X_learn),
                pd.DataFrame(y_learn),
                seq_len,
                keep_percentage
            )

            # VALIDATION SEQUENCES
            # ===================================
            X_val_seq, y_val_seq = utils.create_sequences_uniform(
                pd.DataFrame(X_val),
                pd.DataFrame(y_val),
                seq_len,
                keep_percentage
            )

            # SEQUENCES TEST
            # ========================================================
            X_test_full = np.vstack([
                X_train_scaled[-seq_len:],  # end of 2022
                X_test_scaled               # begining of 2023
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
                "x": X_learn_seq,
                "y": y_learn_seq,
                "validation_data": (X_val_seq, y_val_seq),
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
            df_predicted.loc[df_predicted['index'].str.contains(name_station), 'y'] = y_pred

            # MEMORY CLEANUP
            # ========================================================
            K.clear_session()
            del model

        except Exception as e:
            print(f"Station {name_station} skipped due to error {e}")

    return df_predicted, all_losses