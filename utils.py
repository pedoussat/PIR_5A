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
"""

# --------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# --------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------
# Stations with insufficient data (recent openings) that disrupt training
RECENT_STATIONS = {'P6E', 'BDC', 'W80', 'W14', 'QD6'}

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

def combine_df(df_all_stations, df_qd6):
    # Set 'index' as the actual DataFrame index to allow alignment
    df_predicted_idx = df_all_stations.set_index('index')
    df_qd6_idx = df_qd6.set_index('index')

    # Update the 'y' column specifically for QD6 indices
    df_predicted_idx.loc[df_qd6_idx.index, 'y'] = df_qd6_idx['y']

    # Reset index to return to original format
    df_final = df_predicted_idx.reset_index()

    return df_final

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
# DICTIONARIES CREATION
# --------------------------------------------------------------------------------

def create_station_dict(data):
    """
    Create a dictionary station_name -> DataFrame sorted by date
    """
    data=data.copy()
    # Recreate the mapping dictionary to replace station names with numeric IDs
    X_station = data['station']
    station_mapping = {station: i for i, station in enumerate(X_station.unique())}
    data['station_id'] = data['station'].map(station_mapping)

    # Organize data by station and date
    stations = data['station'].unique()

    # Recreate the dictionnary sorted by date
    df_per_station = {station: data[data['station'] == station].sort_values(by='date') for station in stations}
    return df_per_station

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
    Create randomly shuffled sequences, all possible sequences will be created.
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


def create_sequences_uniform(X, y, seq_len, keep_percentage = 0.25):
    """
    Creates sequences distributed uniformly across the entire time period.
    """
    X_list = []
    y_list = []

    # Calculate the total number of possible sequences
    total_possible = len(X) - seq_len
    
    # Calculate the 'stride' (step) to maintain the desired percentage
    step = int(1 / keep_percentage)
    
    # Generate indices linearly with the calculated step
    # This ensures coverage from the beginning to the end of the time series
    indices = np.arange(0, total_possible, step)

    if len(indices) < 5: # Force a smaller step if we have too few samples
        indices = np.linspace(0, total_possible - 1, num=5, dtype=int)

    for i in indices:
        v = X.iloc[i:(i + seq_len)].values
        X_list.append(v)
        y_list.append(y.iloc[i + seq_len])

    # Convert to numpy arrays
    X_array = np.array(X_list)
    y_array = np.array(y_list)

    # Shuffle the final results
    # Important for model training to avoid chronological bias, 
    # while maintaining the uniform spatial coverage already captured.
    permutation = np.random.permutation(len(X_array))
    
    return X_array[permutation], y_array[permutation]


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

# ---------------------------------------------------------------------------------
# SPLITING
# Call
# df_train, df_test = utils.split_dataset(df, cut_date='2022-06-01')
# Code
# ---------------------------------------------------------------------------------
def split_dataset(df, cut_date='2022-06-01'):
    df = df.copy()
    # Split before and after 2022-05-31
    cut_mid_2022 = pd.to_datetime(cut_date)
    data_train = df[df['date'] < cut_mid_2022]
    data_test = df[df['date'] >= cut_mid_2022]
    return data_train, data_test

# ---------------------------------------------------------------------------------
# EXTRACT RECENT STATIONS
# ------------------------------------------------------------------------------
def filter_stations(data_dict, stations_to_exclude):
    """
    Filters out specific stations from a dictionary of DataFrames.

    Args:
        data_dict (dict): Dictionary where keys are station names and values are DataFrames.
        stations_to_exclude (list): List of station names (strings) to be removed.

    Returns:
        dict: A new dictionary containing only the stations not present in the exclusion list.
    """
    # Use a dictionary comprehension to build the filtered dictionary
    return {
        station: df for station, df in data_dict.items() 
        if station not in stations_to_exclude
    }

# ======================================================================================
# --------------------------------------------------------------------------------------
# GLOBAL PREPARATION PIPELINES


# SUBMISSION
# ======================================================================================
# Call
# df_train, df_test = utils.prepare_submission_data(x_train, y_train, x_test, remove_covid=True)
# --------------------------------------------------------------------------------------
# Code
# --------------------------------------------------------------------------------------
def prepare_submission_data(x_train, y_train, x_test, remove_covid=True):
    """
    Prepares the final dataset for the 2023 competition submission.
    """
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
    df_train= create_station_dict(data_train)
    df_test = create_station_dict(data_test)

    return df_train, df_test

# BACKTEST
# ======================================================================================
# Call
# df_train = utils.prepare_backtest_data(x_train, y_train, remove_covid=True)
# --------------------------------------------------------------------------------------
# Code
# --------------------------------------------------------------------------------------
def prepare_backtest_data(x, y, remove_covid=True):
    """
    Prepares training and backtesting sets with chronological splitting.

    Steps:
    1. Sorts data and aligns features with targets via a unique 'index'.
    2. Optional: Removes COVID-19 period to eliminate anomalous data.
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

    # Create train and test dictionnaries
    df_train = create_station_dict(data)

    return df_train


# DATA PREPARATION FOR PREDICTION ON 2023 EXPLORATION
# Call
# df_established_predicted, df_recent_predicted = utils.prepare_results_data(x_test, y_attempt)
# ======================================================================================
def prepare_results_data(x_test, y_attempt):
    """
    Integrates model predictions with test features for evaluation and visualization.
    Uses 'create_production_dicts' to maintain the separation between established 
    and recent stations in the final output.

    Args:
        x_test (pd.DataFrame): The test features.
        y_attempt (pd.DataFrame): The predicted values aligned by 'index'.

    Returns:
        tuple: (dict_est_predicted, dict_rec_predicted)
    """
    # Ensure chronological consistency
    x_test['date']  = pd.to_datetime(x_test['date'] , format='%Y-%m-%d') #'YYYY-MM-DD'

    # Sort the DataFrame by date
    x_test = x_test.sort_values(by='date').reset_index(drop=False) # keep the index

    # Merge x_test and y_attempt on 'index'
    data_test = pd.merge(x_test, y_attempt, on='index')
    data_test = data_test[['date', 'station', 'index', 'job', 'ferie', 'vacances', 'y']] #reorder columns

    # Create attempt dictionnary
    df_predicted = create_station_dict(data_test)

    return df_predicted

# --------------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------------
# Call
# plot_training_loss(loss, val_loss)
# Code
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
    Plots training, predicted, and true values for a specific station with enlarged fonts.
    """
    # Extracting Data
    df_train_station = df_train[station_name]
    df_test_station = df_test[station_name]
    df_true_station = df_true[station_name]

    # Filter by start date
    start = pd.to_datetime(start_date, format='%Y-%m-%d')
    df_train_station = df_train_station[df_train_station['date'] > start]

    # Initialize Figure
    plt.figure(figsize=(20, 8))

    # Plotting
    plt.plot(df_train_station['date'], 
             df_train_station['y'],
             label='Training Values',
             color='blue',
             alpha=0.4)
    
    plt.plot(df_true_station['date'], 
             df_true_station['y'],
             label='Actual Values',
             color='red',
             alpha=0.7)
    
    plt.plot(df_test_station['date'], 
             df_test_station['y'],
             label='Predicted Values',
             color='green')

    # Font size customization
    TITLE_FONT_SIZE = 22
    LABEL_FONT_SIZE = 16
    LEGEND_FONT_SIZE = 14

    # Labels and Titles
    plt.title(f'Backtest Analysis: Station {station_name}', fontsize=TITLE_FONT_SIZE, pad=20)
    plt.xlabel('Date', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Target Value (y)', fontsize=LABEL_FONT_SIZE)
    
    # Grid for better readability
    plt.grid(True, linestyle=':', alpha=0.6)

    # Enlarged Legend
    # loc='upper left' or 'best'
    plt.legend(fontsize=LEGEND_FONT_SIZE, loc='best', frameon=True, shadow=True)

    plt.tight_layout()
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