"""
utils.py
========
Utility toolbox for the Transilien Passenger Affluence Prediction project.

This module provides essential functions for the end-to-end data pipeline,
from row data cleaning to final submission formatting.
It also includes various functions for visualizing predictions and evaluating their results.

Main Functions Include:
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
import math

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

def combine_df(df_all_stations, df_one_station):
    # Set 'index' as the actual DataFrame index to allow alignment
    df_predicted_idx = df_all_stations.set_index('index')
    df_one_station_idx = df_one_station.set_index('index')

    # Update the 'y' column specifically for QD6 indices
    df_predicted_idx.loc[df_one_station_idx.index, 'y'] = df_one_station_idx['y']

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

def chose_stations(data_dict, stations_to_exclude):
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
        if station in stations_to_exclude
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
    #data_train = data_train[['date', 'station', 'index', 'job', 'ferie', 'vacances', 'y']] #reorder columns

    # Create y_test (fake y)
    y_test = create_y_test(x_test)

    # Merge x_test and y_test on 'index'
    data_test = pd.merge(x_test, y_test, on='index')
    #data_test = data_test[['date', 'station', 'index', 'job', 'ferie', 'vacances', 'y']] #reorder columns

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
    # data = data[['date', 'station', 'index', 'job', 'ferie', 'vacances', 'y']] #reorder columns

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

# DISPLAY LOSS FUNCTION
# --------------------------------------------------------------------------------
# Call
# --------------------------------------------------------------------------------

# Code
# --------------------------------------------------------------------------------
def plot_all_training_losses(all_losses, max_plots=30, max_cols=3):
    """
    Plots training and validation losses for multiple stations.
    Includes a safety cap on the Y-axis to prevent extreme outliers from hiding data.
    """
    stations_to_plot = list(all_losses.items())[:max_plots]
    n_plots = len(stations_to_plot)
    
    if n_plots == 0:
        print("No loss data to plot.")
        return

    cols = min(n_plots, max_cols)
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows))
    
    if n_plots == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten()

    for i, (station_name, history) in enumerate(stations_to_plot):
        ax = axes_list[i]
        
        # 1. Clean data: Replace None or NaN with 0 for plotting safety
        train_loss = np.array(history.get('train', []))
        val_loss = np.array(history.get('val', []))
        
        # 2. Plot lines
        ax.plot(train_loss, label="Train Loss", color="#1f77b4", marker='o', markersize=3, linewidth=1.5)
        ax.plot(val_loss, label="Val Loss", color="#d62728", marker='o', markersize=3, linewidth=1.5)
        
        # 3. SAFETY: Limit the Y-axis scale
        # We calculate the 95th percentile or a small multiple of the median 
        # to ignore the "Exploding Gradient" values like 4e12.
        all_vals = np.concatenate([train_loss, val_loss])
        all_vals = all_vals[np.isfinite(all_vals)] # Remove Infinity/NaN
        
        if len(all_vals) > 0:
            # We set the limit to 1.5 times the 90th percentile to focus on meaningful data
            y_limit = np.percentile(all_vals, 90) * 2
            # Ensure we don't cap it to 0 if all values are very small
            ax.set_ylim(0, max(y_limit, 0.05)) 

        # --- STYLING ---
        ax.set_title(f"Station {station_name}", fontweight='normal', fontsize=16)
        ax.set_xlabel("Epochs", fontsize=16)
        ax.set_ylabel("Loss (MSE)", fontsize=16)
        ax.legend(fontsize=19)
        ax.grid(True, linestyle='--', alpha=0.6)

    for j in range(n_plots, len(axes_list)):
        axes_list[j].axis('off')

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------
# DISPLAY MAPE RESULTS
# --------------------------------------------------------------------------------
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
def show_predictions_results_one_station(df_train, df_test, df_true, station_name, start_date='2015-01-01', end_date='2023-01-01'):
    """
    Plots training, predicted, and true values for a specific station with enlarged fonts.
    """
    # Extracting Data
    df_train_station = df_train[station_name]
    df_test_station = df_test[station_name]
    df_true_station = df_true[station_name]

    # Display Window
    start = pd.to_datetime(start_date, format='%Y-%m-%d')
    end = pd.to_datetime(end_date, format='%Y-%m-%d')

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
    LABEL_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 20

    # Labels and Titles
    plt.title(f'Backtest Analysis: Station {station_name}', fontsize=TITLE_FONT_SIZE, pad=20)
    plt.xlabel('Date', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Target Value (y)', fontsize=LABEL_FONT_SIZE)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=LEGEND_FONT_SIZE, loc='best', frameon=True, shadow=True)
    plt.xlim(start, end)
    plt.tight_layout()
    plt.show()

def show_predictions_results_one_station_enhanced(df_train, df_test, df_true, station_name, 
                                                  start_date='2015-01-01', end_date='2023-01-01'):
    """
    Plots training, predicted, and true values for a specific station with:
    - Holidays as background
    - 'ferie' as red diamonds
    - Sundays as orange triangles
    - Enlarged fonts and grid
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract data
    df_train_station = df_train[station_name]
    df_test_station = df_test[station_name]
    df_true_station = df_true[station_name]

    # Combine all for consistent highlighting
    df_all = pd.concat([df_train_station, df_test_station, df_true_station]).drop_duplicates(subset='date')
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_train_station['date'] = pd.to_datetime(df_train_station['date'])
    df_test_station['date'] = pd.to_datetime(df_test_station['date'])
    df_true_station['date'] = pd.to_datetime(df_true_station['date'])

    # Display window
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    plt.figure(figsize=(12, 8))

    # Highlight vacations as background
    current_start = None
    first_patch = True
    for idx, row in df_all.iterrows():
        if 'vacances' in row and row['vacances'] == 1 and current_start is None:
            current_start = row['date']
        elif (('vacances' not in row or row['vacances'] == 0) or idx == df_all.index[-1]) and current_start is not None:
            current_end = row['date']
            plt.axvspan(current_start, current_end, color='#DDFFDD', alpha=0.3,
                        label='School Holidays' if first_patch else "")
            first_patch = False
            current_start = None

    # Plot series
    #plt.plot(df_train_station['date'], df_train_station['y'], label='Training Values', color='blue', alpha=0.4)
    plt.plot(df_true_station['date'], df_true_station['y'], label='Actual Values', color='red', alpha=0.7)
    plt.plot(df_test_station['date'], df_test_station['y'], label='Predicted Values', color='green')

    # Highlight Sundays
    sundays = df_all[df_all['date'].dt.dayofweek == 6]
    plt.scatter(sundays['date'], sundays['y'], color='orange', marker='^', s=50, label='Sunday')

    # Highlight holidays with red diamonds
    holidays = df_all[df_all['ferie'] == 1]
    plt.scatter(holidays['date'], holidays['y'], color='red', marker='D', s=70, label='\'ferie\'')

    # Font size customization
    TITLE_FONT_SIZE = 22
    LABEL_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 15

    # Labels, titles, grid, legend
    plt.title(f'Station {station_name}: Passenger Attendance (Actual vs Predicted) - LSTM Model', fontsize=TITLE_FONT_SIZE, pad=20)    
    plt.xlabel('Date', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Passenger Attendance', fontsize=LABEL_FONT_SIZE)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=LEGEND_FONT_SIZE, loc='best', frameon=True, shadow=True)
    plt.xlim(start, end)

    # Adjust layout to avoid tight_layout warnings
    plt.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.1)
    plt.show()


# --------------------------------------------------------------------------------
# DISPLAY 2023 PREDICTIONS RESULTS FOR ONE STATION 
# --------------------------------------------------------------------------------
def show_prediction(df_train, df_test, station_name, model_name, start_date='2015-01-01', end_date='2023-07-01'):
    """
    Plot training and predicted values for all stations.
    """
    # Extracting Data
    df_train_station = df_train[station_name]
    df_test_station = df_test[station_name]

    # Display Window
    start = pd.to_datetime(start_date, format='%Y-%m-%d')
    end = pd.to_datetime(end_date, format='%Y-%m-%d')

    # Figure
    plt.figure(figsize=(20,8))

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
    plt.title(f'Predicted Values for station {station_name} with {model_name}', fontsize=20)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.legend(fontsize=20, loc='best', frameon=True, shadow=True)
    plt.xlim(start, end)
    plt.tight_layout()
    plt.show()


def show_prediction3(df_train, df_test, station_name, model_name, start_date='2015-01-01', end_date='2023-07-01'):
    """
    Plot training and predicted values for a station with:
    - holidays as background color
    - 'ferie' and Sundays as markers
    - legend for all
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract Data
    df_train_station = df_train[station_name]
    df_test_station = df_test[station_name]

    # Combine train and test for consistent plotting
    df_all = pd.concat([df_train_station, df_test_station])
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_train_station['date'] = pd.to_datetime(df_train_station['date'])
    df_test_station['date'] = pd.to_datetime(df_test_station['date'])

    # Display window
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    plt.figure(figsize=(12,8))

    # Highlight vacations as continuous background
    current_start = None
    first_patch = True
    for idx, row in df_all.iterrows():
        if row['vacances'] == 1 and current_start is None:
            current_start = row['date']
        elif (row['vacances'] == 0 or idx == df_all.index[-1]) and current_start is not None:
            current_end = row['date']
            plt.axvspan(current_start, current_end, color='#DDFFDD', alpha=0.3,
                        label='School Holidays' if first_patch else "")
            first_patch = False
            current_start = None

    # Plot actual and predicted values
    plt.plot(df_train_station['date'], df_train_station['y'], label='Actual Values', color='blue', marker='o', markersize=4)
    plt.plot(df_test_station['date'], df_test_station['y'], label='Predicted Values', color='green', marker='o', markersize=4)

    # Highlight holidays with red diamonds
    holidays = df_all[df_all['ferie'] == 1]
    plt.scatter(holidays['date'], holidays['y'], color='red', marker='D', s=70, label='\'ferie\'')

    # Highlight Sundays with orange triangles
    sundays = df_all[df_all['date'].dt.dayofweek == 6]
    plt.scatter(sundays['date'], sundays['y'], color='black', marker='H', s=50, label='Sunday')

    # Titles and labels
    plt.title(f'Predicted Values for station {station_name} with {model_name}', fontsize=20)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.legend(fontsize=16, loc='best', frameon=True, shadow=True)
    plt.xlim(start, end)
    plt.tight_layout()
    plt.show()
