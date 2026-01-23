"""
modelsRecent.py
=========

This module contains the deep learning architectures used for predicting 
passenger affluence for the most recent stations
"""

# IMPORTS
# ===============================================================================================================================================================================================
import utils

import matplotlib.pyplot as plt


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
def backtest_model(df_train_per_station, df_test_per_station, df_true_per_station,
               seq_len, units, activation, learning_rate, batch_size, epochs,
               keep_percentage = 0.25,
               early_stop = True,
               features = ['job', 'ferie', 'vacances'],
               architecture = 'rnn'):
    idx=1
    mape_results = []
    all_losses = {}
    sample_size = 3

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

            # LEARNING SEQUENCES
            # ===================================
            X_train_seq, y_train_seq = utils.create_sequences_uniform(
                pd.DataFrame(X_train_scaled),
                pd.DataFrame(y_train_scaled),
                seq_len,
                keep_percentage
            )

            # CHRONOLOGICAL SPLIT ON SEQUENCES
            split = int(len(X_train_seq) * 0.8)

            # On s'assure que ce sont des arrays numpy
            X_learn_seq = np.array(X_train_seq[:split])
            y_learn_seq = np.array(y_train_seq[:split])

            X_val_seq = np.array(X_train_seq[split:])
            y_val_seq = np.array(y_train_seq[split:])

            # VÉRIFICATION DE SÉCURITÉ
            print(f"Shape X_learn: {X_learn_seq.shape}") # Doit être (Samples, 15, 3)
            print(f"Shape X_val: {X_val_seq.shape}")     # Doit être (Samples, 15, 3)


            # SEQUENCES TEST
            # ========================================================
            X_test_full = np.vstack([
                X_train_scaled[-seq_len:],  # end of 2022
                X_test_scaled               # begining of 2023
            ])

            X_test_seq = []
            for i in range(len(X_test)):
                X_test_seq.append(X_test_full[i:i+seq_len])

            X_test_seq = np.array(X_test_seq)

            # MODEL CONSTRUCTION
            # ========================================================
            # 
            input_shape = (seq_len, len(features))
            
            if architecture == 'rnn':
                model = Sequential([
                    Input(shape=input_shape),
                    SimpleRNN(units=units, activation=activation),
                    Dense(1)
                ])
            elif architecture == 'lstm':
                model = Sequential([
                    Input(shape=input_shape),
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
            mape_results.append({"station": name_station, "MAPE": score})

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
# df_predicted, all_losses = modelsRecent.submission_model(df_train, df_test, df_predicted, seq_len, units, activation, learning_rate, batch_size, epochs, early_stop = True, features = ['job', 'ferie', 'vacances'])

# Code
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def submission_model(df_per_station_train, df_per_station_test, df_predicted,
                     seq_len, units, activation, learning_rate, batch_size,
                     epochs, keep_percentage=0.5, early_stop = True,
                     features = ['job', 'ferie', 'vacances'],
                     architecture = 'rnn'):
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
            # 
            input_shape = (seq_len, len(features))
            
            if architecture == 'rnn':
                model = Sequential([
                    Input(shape=input_shape),
                    SimpleRNN(units=units, activation=activation),
                    Dense(1)
                ])
            elif architecture == 'lstm':
                model = Sequential([
                    Input(shape=input_shape),
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

            # 4. FIX: THE ASSIGNMENT ERROR
            # We use .flatten() to convert (N, 1) to (N,)
            mask = df_predicted['index'].str.contains(name_station)
            
            # Safety check: ensure lengths match
            if len(df_predicted.loc[mask]) == len(y_pred):
                df_predicted.loc[mask, 'y'] = y_pred.flatten()
            else:
                print(f"Warning: Length mismatch for {name_station}. Expected {len(y_pred)} rows.")

            # ASSIGNMENT
            # =========================
            #df_predicted.loc[df_predicted['index'].str.contains(name_station), 'y'] = y_pred

            # MEMORY CLEANUP
            # ========================================================
            K.clear_session()
            del model

        except Exception as e:
            print(f"Station {name_station} skipped due to error {e}")

    return df_predicted, all_losses

# ========================================================================================



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