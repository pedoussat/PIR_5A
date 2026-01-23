"""
AR_models.py
========
Autoregressive LSTM models
"""

# IMPORTS
# ===============================================================================================================================================================================================
import utils

import pandas as pd
import numpy as np

import os
import keras.backend as K

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard


# ROLLING PREDICTION FUNCTION
# ===============================================================================================================================================================================================
def predict_rolling(model, last_sequence, future_calendar_scaled):
    """
    Predicts step-by-step by re-injecting the previous prediction back into the input window.
    - last_sequence: (seq_len, n_features) - The final part of the scaled training set.
    - future_calendar_scaled: (n_test_days, n_features - 1) - known future features = x_test
    """
    predictions = []
    # Initialize the sliding window with the last known data points
    current_window = last_sequence.copy()

    for i in range(len(future_calendar_scaled)):
        # 1. Predict the next day
        # np.newaxis converts (seq_len, n_features) to (1, seq_len, n_features) for the LSTM batch requirement
        y_pred_s = model.predict(current_window[np.newaxis, :, :], verbose=0)[0][0]
        
        # Prevent the model from re-injecting negative values
        y_pred_s = max(0, y_pred_s) 
        predictions.append(y_pred_s)

        # 2. Construct the next input row
        # Combine the new prediction with the known calendar features for that specific day
        next_row = np.hstack(([y_pred_s], future_calendar_scaled[i]))

        # 3. Slide the window (The "Rolling" part)
        # Drop the first (oldest) row and append the 'next_row' at the end
        current_window = np.vstack((current_window[1:], next_row))

    return np.array(predictions)

# BACKTEST: AUTOREGRESSIVE LSTM MODEL EXPLORATION RUN FOR ALL STATIONS
# ===============================================================================================================================================================================================
# Call
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# results, mape, histories = AR_models.lstm_backtest_autoregressive(
#    sample_train, 
#    copy.deepcopy(sample_test), 
#    sample_test_true, 
#    seq_len, units, activation, learning_rate, batch_size
#)

# Code
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def lstm_backtest_autoregressive(sample_train, sample_test, sample_test_true,
                                 seq_len, units, activation, learning_rate, batch_size,
                                 keep_percentage = 0.25,
                                 features=['y', 'job', 'ferie', 'vacances']):
    """
    Performs a backtest by training an independent Autoregressive LSTM model for each station.
    """
    mape_results = []
    all_histories = {}
    
    # Identify exogenous variables (everything except the target 'y')
    exo_features = [f for f in features if f != 'y']
    
    for idx, (name_station, df_train) in enumerate(sample_train.items(), 1):
        try:
            print(f"[{idx}/{len(sample_train)}] Processing Station: {name_station}")
            df_test = sample_test[name_station]
            
            # 1 --- SCALING ---
            # Using two separate scalers prevents binary/calendar features from 
            # interfering with the normalization statistics of the ridership data.
            scaler_X = MinMaxScaler() 
            scaler_y = MinMaxScaler() 
            y_train_scaled = scaler_y.fit_transform(df_train[['y']])
            X_exo_scaled = scaler_X.fit_transform(df_train[exo_features])

            # Recombine into a single matrix: [y_scaled, exo1_scaled, exo2_scaled, ...]
            X_train_full_scaled = np.hstack((y_train_scaled, X_exo_scaled))

            # 2 --- SEQUENCE PREPARATION ---
            # Creates the sliding windows (X) and their corresponding next-step targets (y)
            X_train_seq, y_train_seq = utils.create_sequences_uniform(
                pd.DataFrame(X_train_full_scaled), pd.DataFrame(y_train_scaled),
                seq_len, keep_percentage=keep_percentage
            )

            # 3 --- CHRONOLOGICAL SPLIT ---
            # We take the first 80% of sequences for training and the last 20% for validation.
            split_idx = int(len(X_train_seq) * 0.8)

            X_learn = X_train_seq[:split_idx]
            y_learn = y_train_seq[:split_idx] 

            X_val = X_train_seq[split_idx:]
            y_val = y_train_seq[split_idx:] 

            # 4 --- MODEL DEFINITION ---
            model = Sequential([
                Input(shape=(seq_len, len(features))),
                LSTM(units=units, activation=activation),
                Dense(1),
                # Safety floor to ensure non-negative metro ridership
                LeakyReLU(negative_slope=0.01) 
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
            
            # Setup monitoring
            log_dir = os.path.join("logs", "fit", name_station)
            tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) 
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            history = model.fit(
                X_learn, y_learn, 
                validation_data=(X_val, y_val),
                epochs=15, 
                batch_size=batch_size, 
                callbacks=[early_stop, tb_callback],
                verbose=0
            )

            all_histories[name_station] = {
                'train': history.history['loss'],
                'val': history.history['val_loss']
            }

            # 5 --- AUTOREGRESSIVE PREDICTION (Rolling Forecast) ---
            test_calendar_scaled = scaler_X.transform(df_test[exo_features])
            last_train_window = X_train_full_scaled[-seq_len:]

            y_pred_scaled = predict_rolling(model, last_train_window, test_calendar_scaled)

            # 6 --- EVALUATION ---
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
            y_true = sample_test_true[name_station]['y'].values
            
            score = mean_absolute_percentage_error(y_true, y_pred)
            mape_results.append({"station": name_station, "MAPE": score})
            
            sample_test[name_station]['y'] = y_pred.flatten()

            # Memory Management
            K.clear_session()

        except Exception as e:
            print(f"Error for station {name_station}: {e}")

    return sample_test, mape_results, all_histories

# PREDICTION ON 2023: AUTOREGRESSIVE LSTM MODEL RUN FOR ALL STATIONS
# ===============================================================================================================================================================================================
# Call
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# results, histories = AR_models.submission_lstm(
#    sample_train, 
#    copy.deepcopy(sample_test), 
#    sample_test_true, 
#    seq_len, units, activation, learning_rate, batch_size
#)

# Code
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def submission_lstm(sample_train, sample_test, predictions,
                                 seq_len, units, activation, learning_rate, batch_size,
                                 features=['y', 'job', 'ferie', 'vacances']):

    all_histories = {}
    
    # Identify exogenous variables (everything except the target 'y')
    exo_features = [f for f in features if f != 'y']
    
    for idx, (name_station, df_template) in enumerate(predictions.items(), 1):        
        try:
            print(f"[{idx}/{len(sample_train)}] Processing Station: {name_station}")
            df_test = sample_test[name_station]
            df_train = sample_train[name_station]
            
            # 1 --- SCALING ---
            # Using two separate scalers prevents binary/calendar features from 
            # interfering with the normalization statistics of the ridership data.
            scaler_X = MinMaxScaler() 
            scaler_y = MinMaxScaler() 
            y_train_scaled = scaler_y.fit_transform(df_train[['y']])
            X_exo_scaled = scaler_X.fit_transform(df_train[exo_features])

            # Recombine into a single matrix: [y_scaled, exo1_scaled, exo2_scaled, ...]
            X_train_full_scaled = np.hstack((y_train_scaled, X_exo_scaled))

            # 2 --- SEQUENCE PREPARATION ---
            # Creates the sliding windows (X) and their corresponding next-step targets (y)
            X_train_seq, y_train_seq = utils.create_sequences_uniform(
                pd.DataFrame(X_train_full_scaled), pd.DataFrame(y_train_scaled),
                seq_len, keep_percentage=0.25
            )

            # 3 --- CHRONOLOGICAL SPLIT ---
            # We take the first 80% of sequences for training and the last 20% for validation.
            split_idx = int(len(X_train_seq) * 0.8)

            X_learn = X_train_seq[:split_idx]
            y_learn = y_train_seq[:split_idx] 

            X_val = X_train_seq[split_idx:]
            y_val = y_train_seq[split_idx:] 

            # 4 --- MODEL DEFINITION ---
            model = Sequential([
                Input(shape=(seq_len, len(features))),
                LSTM(units=units, activation=activation),
                Dense(1),
                # Safety floor to ensure non-negative metro ridership
                LeakyReLU(negative_slope=0.01) 
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
            
            # Setup monitoring
            log_dir = os.path.join("logs", "fit", name_station)
            tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) 
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            history = model.fit(
                X_learn, y_learn, 
                validation_data=(X_val, y_val),
                epochs=15, 
                batch_size=batch_size, 
                callbacks=[early_stop, tb_callback],
                verbose=0
            )

            all_histories[name_station] = {
                'train': history.history['loss'],
                'val': history.history['val_loss']
            }

            # 5 --- AUTOREGRESSIVE PREDICTION (Rolling Forecast) ---
            test_calendar_scaled = scaler_X.transform(df_test[exo_features])
            last_train_window = X_train_full_scaled[-seq_len:]

            y_pred_scaled = predict_rolling(model, last_train_window, test_calendar_scaled)

            # 6 --- ASSIGNEMENT ---
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
            predictions[name_station]['y'] = y_pred.flatten()

            # Memory Management
            K.clear_session()

        except Exception as e:
            print(f"Error for station {name_station}: {e}")

    return predictions, all_histories