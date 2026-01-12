"""
utils.py
========
Utility toolbox for the Transilien Passenger Affluence Prediction project.

This module provides essential functions for the end-to-end data pipeline,
from row data cleaning to final submission formatting.
It also includes various functions for visualizing predictions and evaluating their results.

CORE CAPABILITIES:
-----------------
1. Data Preprocessing & Cleaning:
   - Chronological sorting and datetime standardization.
   - COVID-19 anomaly removal: Filters out the 2020-2021 period to prevent 
     the model from learning non-representative passenger behaviors.
   - Station-based partitioning: Organizes global data into dictionaries 
     mapped by station names for local model training.

2. Sequence Engineering:
   - Sliding Window Generation: Transforms DataFrames into 3D NumPy arrays 
     (Samples, Time Steps, Features) required by LSTM architectures.
   - Random Shuffling: Supports non-chronological validation by shuffling 
     temporal sequences while preserving internal time-step order.

3. Visualization & Analytics:
   - Learning Curve Tracking: Plots training vs. validation loss to 
     diagnose convergence, underfitting, or overfitting.
   - Performance Assessment: Visual comparison of "True vs. Predicted" 
     values and MAPE (Mean Absolute Percentage Error) calculation.

4. Submission Pipeline:
   - Reformatting: Ensures the final predictions match the exact index 
     requirements of the competition submission format.

USAGE:
------
Imported as 'utils' in 'models.py' or main notebooks to maintain a clean 
separation between data logic and model architecture.
"""

# --------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------------------------------------
# DATA PREPARATION FUNCTIONS
# --------------------------------------------------------------------------------
# Creation of y_test
def create_y_test(x_test):
    """
    Generate a placeholder DataFrame for test targets.
    
    Creates a basic 'y' column filled with a constant value (1000.0) 
    mapped to the unique 'index' of each row. This is used to initialize 
    the test structure before actual model predictions.

    Args:
        x_test (pd.DataFrame): The test feature DataFrame containing an 'index' column.

    Returns:
        pd.DataFrame: A DataFrame with 'index' and a placeholder 'y' column.
    """
    y_test = pd.DataFrame()
    y_test['index'] = x_test['index']
    y_test['y'] = 1000.0
    return y_test

def create_y_attempt(x_test):
    """
    Prepare a sorted placeholder DataFrame for a submission attempt.
    
    This function ensures the input features are chronologically ordered by 'date' 
    before generating a dummy prediction column. It preserves the original 
    row indexing to maintain traceability.

    Args:
        x_test (pd.DataFrame): The test feature DataFrame to be sorted and processed.

    Returns:
        pd.DataFrame: A sorted DataFrame containing the 'index' and a constant 'y' value.
    """
    x_test['date']  = pd.to_datetime(x_test['date'] , format='%Y-%m-%d') #'YYYY-MM-DD'
    x_test = x_test.sort_values(by='date').reset_index(drop=True)
    y_test = pd.DataFrame()
    y_test['index'] = x_test['index']
    y_test['y'] = 1000.0
    return y_test

def remove_covid_period (data, start_date = '2020-03-01', end_date = '2021-05-31'):
    """
    Remove COVID-19 period from the dataset.
    """
    # Convert to pandas datetime format
    start = pd.to_datetime(start_date, format='%Y-%m-%d')
    end = pd.to_datetime(end_date, format='%Y-%m-%d')
    # Filtering
    mask = (data['date'] < start) | (data['date'] > end)
    data = data[mask].reset_index(drop=True)
    return data


def creat_station_dict(data):
    """
    Create a dictionary station_name -> DataFrame sorted by date
    """
    # Recreate the mapping dictionary to replace station names with numeric IDs
    X_station = data['station']
    station_mapping = {station: i for i, station in enumerate(X_station.unique())}
    data['station_id'] = data['station'].map(station_mapping)

    # Organize data by station and date
    stations = data['station'].unique()

    # Recreate the dictionnary sorted by date
    df_per_station = {station: data[data['station'] == station].sort_values(by='date') for station in stations}
    return df_per_station

def reorder_for_submission(y_attempt, y_example):
    """
    Realigns predictions to match the official submission index order.

    Args:
        y_attempt (pd.DataFrame): The generated predictions with unique 'index'.
        y_example (pd.DataFrame): The official template/sample submission file.

    Returns:
        pd.DataFrame: Sorted predictions ready for CSV export.
    """    
    y_test_sorted = (y_attempt.set_index('index').loc[y_example['index']].reset_index())
    return y_test_sorted

# --------------------------------------------------------------------------------
# SEQUENCE CREATION
# --------------------------------------------------------------------------------
def create_sequences(X, y, seq_len):
    """
    Create sliding-window sequences.
    """
    X_list = [] 
    y_list = []

    for i in range(len(X) - seq_len):
        v = X.iloc[i:(i + seq_len)].values
        X_list.append(v)
        y_list.append(y.iloc[i + seq_len])

    return np.array(X_list), np.array(y_list)


def create_sequences_random(X, y, seq_len):
    """
    Create randomly shuffled sequences.
    """
    X_list = []
    y_list = []

    # Create all possible starting indices for sequences
    indices = np.arange(len(X) - seq_len)

    # Suffle the indices to generate sequences in random order
    np.random.shuffle(indices)

    for i in indices:
        v = X.iloc[i:(i + seq_len)].values
        X_list.append(v)
        y_list.append(y.iloc[i + seq_len])

    return np.array(X_list), np.array(y_list)


# ---------------------------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------------------------
def sample_stations(df_per_station_train, n, seed=None):
    """
    Randomly sample n statios from a station dictionnary.
    """
    if seed is not None:
        random.seed(seed)

    sampled_keys = random.sample(
        list(df_per_station_train.keys()),
        k=min(n, len(df_per_station_train))
    )

    return {k: df_per_station_train[k] for k in sampled_keys}


# DATA PREPARATION FOR PREDICTION ON 2023
# ======================================================================================
def prepare_data(x_train, y_train, x_test, remove_covid=True):
    # Ensure chronological consistency
    x_train['date'] = pd.to_datetime(x_train['date'], format='%Y-%m-%d') #'YYYY-MM-DD'
    x_test['date']  = pd.to_datetime(x_test['date'] , format='%Y-%m-%d') #'YYYY-MM-DD'

    # Sort the DataFrame by date
    x_train = x_train.sort_values(by='date').reset_index(drop=False) # keep the index
    x_test = x_test.sort_values(by='date').reset_index(drop=False) # keep the index

    # Create the 'index' variable in x_train
    x_train['index'] = x_train['date'].dt.strftime('%Y-%m-%d').str.cat(x_train['station'], sep='_')

    # Merge x and y_train on 'index'
    data_train = pd.merge(x_train, y_train, on='index')
    data_train = data_train[['date', 'station', 'index', 'job', 'ferie', 'vacances', 'y']] #reorder columns

    # Create y_test (fake y)
    y_test = create_y_test(x_test)

    # Merge x_test and y_test on 'index'
    data_test = pd.merge(x_test, y_test, on='index')
    data_test = data_test[['date', 'station', 'index', 'job', 'ferie', 'vacances', 'y']] #reorder columns

    # Remove Covid Period from x_train
    if remove_covid:
        data_train = remove_covid_period(data_train)

    # Create train and test dictionnaries
    df_per_station_train = creat_station_dict(data_train)
    df_per_station_test = creat_station_dict(data_test)

    return df_per_station_train, df_per_station_test


# DATA PREPARATION FOR PREDICTION ON LAST 2022
# ======================================================================================
def prepare_train_data(x, y, remove_covid=True):
    """
    Prepares training and backtesting sets with chronological splitting.

    Steps:
    1. Sorts data and aligns features with targets via a unique 'index'.
    2. Optional: Removes COVID-19 period to eliminate anomalous data.
    3. Split: Uses June 2022 as a cut-off to create a 'pseudo-future' test set.
    4. Mapping: Groups data into station-specific dictionaries for local modeling.

    Args:
        x, y (pd.DataFrame): Input features and targets.
        remove_covid (bool): Toggle for filtering out 2020-2021 noise.

    Returns:
        tuple: (train_dict, test_dict) partitioned by station.
    """
    # Ensure chronological consistency
    x['date'] = pd.to_datetime(x['date'], format='%Y-%m-%d') #'YYYY-MM-DD'

    # Sort the DataFrame by date
    x = x.sort_values(by='date').reset_index(drop=False) # keep the index

    # Create the 'index' variable in x_train
    x['index'] = x['date'].dt.strftime('%Y-%m-%d').str.cat(x['station'], sep='_')

    # Merge x and y_train on 'index'
    data = pd.merge(x, y, on='index')
    data = data[['date', 'station', 'index', 'job', 'ferie', 'vacances', 'y']] #reorder columns

    # Remove Covid Period from x_train
    if remove_covid:
        data = remove_covid_period(data)

    # Split before and after 2022-05-31
    cut_mid_2022 = pd.to_datetime('2022-06-01')
    data_train = data[data['date'] < cut_mid_2022]
    data_test = data[data['date'] >= cut_mid_2022]

    # Create train and test dictionnaries
    df_per_station_train = creat_station_dict(data_train)
    df_per_station_test = creat_station_dict(data_test)

    return df_per_station_train, df_per_station_test



# DATA PREPARATION FOR PREDICTION ON 2023 EXPLORATION
# ======================================================================================
def prepare_attempt_data(x_test, y_attempt):
    # Ensure chronological consistency
    x_test['date']  = pd.to_datetime(x_test['date'] , format='%Y-%m-%d') #'YYYY-MM-DD'

    # Sort the DataFrame by date
    x_test = x_test.sort_values(by='date').reset_index(drop=False) # keep the index

    # Merge x_test and y_attempt on 'index'
    data_test = pd.merge(x_test, y_attempt, on='index')
    data_test = data_test[['date', 'station', 'index', 'job', 'ferie', 'vacances', 'y']] #reorder columns

    # Create attempt dictionnary
    df_per_station_test = creat_station_dict(data_test)

    return df_per_station_test


# --------------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------------
# Displaying the loss functions
def plot_training_loss(loss, val_loss, title="Training and Validation Loss"):
    """
    Plot the training and validation losses functions
    """
    # Figure
    plt.figure(figsize=(10,5))
    plt.plot(loss, label="Training Loss", color="blue")
    plt.plot(val_loss, label="Validation Loss", color="red")
    
    # Titles
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def show_mape_results(mape_results):
    """
    Print MAPE values per station and the overall average MAPE
    """
    df_mape = pd.DataFrame(mape_results)
    mape_mean = df_mape["MAPE"].mean()

    # Display
    print("MAPE per station:")
    print(df_mape)
    print(f"\nAverage MAPE over all stations: {mape_mean:.4f}")


# --------------------------------------------------------------------------------
# DISPLAY PREDICTIONS RESULTS FOR ONE STATION WHEN YOU KNOW THE TRUE VALUES
# --------------------------------------------------------------------------------
def show_predictions_results_one_station(df_train, df_test, df_true, station_name, start_date='2015-01-01'):
    """
    Plot training, predicted and true values for one station.
    """
    # Extracting Data
    df_train_station = df_train[station_name]
    df_test_station = df_test[station_name]
    df_true_station = df_true[station_name]

    # Display Window
    start = pd.to_datetime(start_date, format='%Y-%m-%d')
    df_train_station= df_train_station[df_train_station['date'] > start]

    # Figure
    plt.figure(figsize=(20,6))

    plt.plot(df_train_station['date'],  # TRAIN
             df_train_station['y'],
             label='Training Values',
             color='blue',
             alpha=0.5)
    
    plt.plot(df_test_station['date'],  # TEST
             df_test_station['y'],
             label='Predicted Values',
             color='green')

    plt.plot(df_true_station['date'],  # TRUE
             df_true_station['y'],
             label='Actual Values',
             color='black',
             alpha=0.5)

    # Titles
    plt.title(f'Actual vs Predicted Values for station {station_name}')
    plt.xlabel('Data Point Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def show_prediction(df_train, df_test, station_name, model_name, start_date='2015-01-01'):
    """
    Plot training and predicted values for all stations.
    """
    # Extracting Data
    df_train_station = df_train[station_name]
    df_test_station = df_test[station_name]

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