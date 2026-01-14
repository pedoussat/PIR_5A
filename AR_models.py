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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, SimpleRNN, LeakyReLU
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, TensorBoard



def predict_rolling(model, last_sequence, future_calendar_scaled):
    """
    Prédit pas à pas en réinjectant la prédiction précédente dans la fenêtre.
    - last_sequence : (seq_len, n_features) - La fin du train scaled.
    - future_calendar_scaled : (n_jours_test, n_features - 1) - Le calendrier du test scaled.
    """
    predictions = []
    current_window = last_sequence.copy()

    for i in range(len(future_calendar_scaled)):
        # 1. Prédire le jour suivant (shape attendue par LSTM: [1, seq_len, n_features])

        y_pred_s = model.predict(current_window[np.newaxis, :, :], verbose=0)[0][0]
        y_pred_s = max(0, y_pred_s) # Empêche la réinjection de valeurs négatives
        predictions.append(y_pred_s)

        # 2. Préparer la ligne suivante : [y_pred, job, ferie, vacances]
        next_row = np.hstack(([y_pred_s], future_calendar_scaled[i]))

        # 3. Faire glisser la fenêtre (Rolling)
        current_window = np.vstack((current_window[1:], next_row))

    return np.array(predictions)


# BACKTEST : AUTOREGRESSIVE LSTM MODEL EXPLORATION RUN FOR ALL STATIONS (prediction over 2022)
# ===============================================================================================================================================================================================
# Call
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# sample_test_AR, mape_results_AR, all_histories_AR = AR_models.lstm_backtest_autoregressive(sample_train, sample_test, sample_test_true, sample_size, seq_len, units, activation, learning_rate, batch_size, epochs)

# Code
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def lstm_backtest_autoregressive(sample_train, sample_test, sample_test_true,
                                 seq_len, units, activation, learning_rate, batch_size,
                                 epochs, features=['y', 'job', 'ferie', 'vacances']):
    mape_results = []
    all_histories = {}
    
    for idx, (name_station, df_train) in enumerate(sample_train.items(), 1):
        try:
            print(f"[{idx}/{len(sample_train)}] Station: {name_station}")
            df_test = sample_test[name_station]
            
            # --- SCALING ---
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_train_scaled = scaler_X.fit_transform(df_train[features])
            y_train_scaled = scaler_y.fit_transform(df_train[['y']])


            # --- ENTRAINEMENT ---
            X_train_seq, y_train_seq = utils.create_sequences_uniform(
                pd.DataFrame(X_train_scaled), pd.DataFrame(y_train_scaled),
                seq_len, keep_percentage=0.50
            )

            # --- RANDOMIZED VALIDATION SPLIT ---
            # shuffle=True permet de prendre des séquences au hasard partout dans l'historique
            X_t, X_v, y_t, y_v = train_test_split(
                X_train_seq, y_train_seq, test_size=0.2, shuffle=True, random_state=42
            )

            # --- MODEL & EARLY STOPPING ---
            model = Sequential([
                Input(shape=(seq_len, len(features))),
                LSTM(units=units, activation=activation),
                Dense(1),
                LeakyReLU(negative_slope=0.01) # Au lieu de activation='relu'
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
            

            #Configuration de TensorBoard
            log_dir = os.path.join("logs", "fit", name_station)
            tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) # histogram_freq=1 active le monitoring des poids

            # Le EarlyStopping protège contre l'overfitting sur le set d'entraînement
            early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
            

            history = model.fit(
                X_t, y_t, 
                validation_data=(X_v, y_v),
                epochs=epochs, 
                batch_size=batch_size, 
                callbacks=[early_stop, tb_callback],
                verbose=0
            )

            all_histories[name_station] = history.history

            # --- PREDICTION AUTOREGRESSIVE ---
            # A. On prépare le calendrier du test (sans la colonne 'y')
            # Le [:, 1:] permet de ne garder que job, ferie, vacances après scaling
            test_calendar_scaled = scaler_X.transform(df_test[features])[:, 1:] 
            
            # B. On prend la dernière fenêtre connue du train
            last_train_window = X_train_scaled[-seq_len:]

            # C. Appel à predict_rolling
            y_pred_scaled = predict_rolling(model, last_train_window, test_calendar_scaled)

            # --- EVALUATION ---
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
            y_true = sample_test_true[name_station]['y'].values
            
            score = mean_absolute_percentage_error(y_true, y_pred)
            mape_results.append({"station": name_station, "MAPE": score})
            sample_test[name_station]['y'] = y_pred.flatten()

            K.clear_session()

        except Exception as e:
            print(f"Error for station {name_station}: {e}")

    return sample_test, mape_results, all_histories