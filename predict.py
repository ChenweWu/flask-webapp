import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import base64
from io import BytesIO
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats
import numpy as np
import sklearn.metrics
import io
import base64
import re
from scipy import stats
# You can use Matplotlib instead of Plotly for viualization by simply replacing `optuna.visualization` with
# `optuna.visualization.matplotlib` in the following examples.

from PIL import Image
from io import BytesIO


import xgboost as xgb
def process_data(file):
    merged_dataframe = file
    merged_dataframe['ds'] = 0
    data_map = {
        1: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "No Failure", "Nomenclature": "No failure"},
        2: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "No Failure", "Nomenclature": "No failure"},
        3: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "FL corner torque loss", "Nomenclature": "Parimeter Bolt"},
        4: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "FR torque loss", "Nomenclature": "Parimeter Bolt"},
        5: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "RL torque loss", "Nomenclature": "Parimeter Bolt"},
        6: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "RR torque loss", "Nomenclature": "Parimeter Bolt"},
        7: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "right middle torque loss", "Nomenclature": "Parimeter Bolt"},
        8: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "front middle torque loss", "Nomenclature": "Parimeter Bolt"},
        9: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "front harness connector", "Nomenclature": "Connector Bolt"},
        10: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "top harness connector", "Nomenclature": "Connector Bolt"},
        11: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "no top cover", "Nomenclature": "No failure"},
        12: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "no top cover internal structure 4 left bolt torque loss", "Nomenclature": "Module Mount Bolt"},
        13: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "with top cover internal structure 4 left bolt torque loss", "Nomenclature": "Module Mount Bolt"},
        14: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "no top cover internal beam 6 bolt torque loss", "Nomenclature": "Transverse Beam Bolt"},
        15: {"Vibration Level": [1, 2, 3, 4], "Number of Sensors": 2, "Failure": "with top cover internal beam left bolt torque loss", "Nomenclature": "Transverse Beam Bolt"},
    }

    device_name_mapping = {
        'e8:cb:ed:5a:4c:83': 0,
        'e8:cb:ed:5a:4c:24': 1
    }

    # Apply the mapping to the 'device_name' column
    merged_dataframe['device_mapped'] = merged_dataframe['Device name'].map(device_name_mapping)
    column_rename_mapping = {
        'X-axis vibration speed(mm/s)': 'Xs',
        'Y-axis vibration speed(mm/s)': 'Ys',
        'Z-axis vibration speed(mm/s)': 'Zs',
        'X-axis vibration displacement(um)': 'Xd',
        'Y-axis vibration displacement(um)': 'Yd',
        'Z-axis vibration displacement(um)': 'Zd'
    }

    # Rename the columns
    merged_dataframe.rename(columns=column_rename_mapping, inplace=True)
    merged_dataframe_0 = merged_dataframe[merged_dataframe.device_mapped ==0].copy()
    merged_dataframe_1 = merged_dataframe[merged_dataframe.device_mapped ==1].copy()
    def iqr(series):
        """
        Calculate the interquartile range (IQR) of a pandas Series.

        Parameters:
        - series: Pandas Series containing the data.

        Returns:
        - iqr: The interquartile range of the data.
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return iqr

    def variance(series):
        """
        Calculate the variance of a pandas Series.

        Parameters:
        - series: Pandas Series containing the data.

        Returns:
        - variance: The variance of the data.
        """
        variance = series.var()
        return variance


    def median(series):
        """
        Calculate the median of a pandas Series.

        Parameters:
        - series: Pandas Series containing the data.

        Returns:
        - median: The median of the data.
        """
        median = series.median()
        return median

    # Define your statistical functions
    def rms(series):
        return np.sqrt(np.mean(np.square(series)))

    def crest_factor(series):
        return np.max(np.abs(series)) / rms(series)

    def form_factor(series):
        return rms(series) / np.mean(np.abs(series))

    # Function to convert time strings to datetime objects
    def convert_to_datetime(time_str):
        time_str = time_str.lstrip()
        return datetime.strptime(time_str, '%H:%M:%S.%f')


    # Assuming merged_dataframe is your DataFrame
    # Convert the 'Time' column to datetime format
    merged_dataframe_0['Time'] = merged_dataframe_0['Time'].apply(convert_to_datetime)

    # Set 'Time' as the index
    merged_dataframe_0.set_index('Time', inplace=True)

    # Define a 1-second interval
    interval = '1S'

    # Initialize an empty DataFrame to store results
    aggregated_data_0 = pd.DataFrame()

    # Define your continuous columns
    continuous_columns = ['Xd', 'Yd', 'Zd','Xs',
        'Ys', 'Zs']  # Replace with your actual column names

    # Iterate over each datasource
    for datasource, group in merged_dataframe_0.groupby('ds'):
        # Resampling by second
        resampled = group.resample(interval)

        # Define aggregation dictionary
        agg_dict = {col: ['max', 'min', 'mean', 'std', rms, scipy.stats.skew, scipy.stats.kurtosis, crest_factor, form_factor, median,variance,iqr] for col in continuous_columns}

        # Calculate statistics for each interval
        stats = resampled.agg(agg_dict)

        # Flatten MultiIndex columns and create new column names
        stats.columns = ['{}_{}'.format(col[0], col[1]) for col in stats.columns]

        # Add additional information
        stats['Interval Length (seconds)'] = 1  # As the interval is 1 second
        stats['Data Points Count'] = resampled.count()[continuous_columns[0]]
        stats['ds'] = datasource
        # Assuming 'Defect', 'sub_defect', 'meta_defect' are columns in the original group
        # stats['Failure'] = group['Failure'].iloc[0]
        # stats['Nomenclature'] = group['Nomenclature'].iloc[0]
        # stats['meta_defect'] = group['meta_defect'].iloc[0]
        # stats['device_mapped'] = group['device_mapped'].iloc[0]
    #     stats['vb'] = group['Vibration Level'].iloc[0]
        # Append to the aggregated data DataFrame
        aggregated_data_0 = aggregated_data_0.append(stats)
        
    # 			device_mapped
    # Resetting index to make Time and Datasource regular columns
    aggregated_data_0.reset_index(inplace=True)


    merged_dataframe_1['Time'] = merged_dataframe_1['Time'].apply(convert_to_datetime)

    # Set 'Time' as the index
    merged_dataframe_1.set_index('Time', inplace=True)

    # Define a 1-second interval
    interval = '1S'

    # Initialize an empty DataFrame to store results
    aggregated_data_1 = pd.DataFrame()

    # Define your continuous columns
    continuous_columns = ['Xd', 'Yd', 'Zd','Xs',
        'Ys', 'Zs']  # Replace with your actual column names

    # Iterate over each datasource
    for datasource, group in merged_dataframe_1.groupby('ds'):
        # Resampling by second
        resampled = group.resample(interval)

        # Define aggregation dictionary
        agg_dict = {col: ['max', 'min', 'mean', 'std', rms, scipy.stats.skew, scipy.stats.kurtosis, crest_factor, form_factor, median,variance,iqr] for col in continuous_columns}

        # Calculate statistics for each interval
        stats = resampled.agg(agg_dict)

        # Flatten MultiIndex columns and create new column names
        stats.columns = ['{}_{}'.format(col[0], col[1]) for col in stats.columns]

        # Add additional information
        stats['Interval Length (seconds)'] = 1  # As the interval is 1 second
        stats['Data Points Count'] = resampled.count()[continuous_columns[0]]
        stats['ds'] = datasource
        # Assuming 'Defect', 'sub_defect', 'meta_defect' are columns in the original group
        # stats['Failure'] = group['Failure'].iloc[0]
        # stats['Nomenclature'] = group['Nomenclature'].iloc[0]
        # stats['meta_defect'] = group['meta_defect'].iloc[0]
        stats['device_mapped'] = group['device_mapped'].iloc[0]
    #     stats['vb'] = group['Vibration Level'].iloc[0]
        # Append to the aggregated data DataFrame
        aggregated_data_1 = aggregated_data_1.append(stats)
        
    # 			device_mapped
    # Resetting index to make Time and Datasource regular columns
    aggregated_data_1.reset_index(inplace=True)
    print(aggregated_data_0.columns.values)
    print(aggregated_data_1.columns.values)
    merged_data = pd.merge(aggregated_data_0, aggregated_data_1, on='Time', how='outer')

    # Sort the merged DataFrame by the 'Time' column if needed
    merged_data.sort_values(by='Time', inplace=True)
    merged_data =merged_data.dropna()
    return merged_data

def load_model():
    # Load your XGBoost model and make predictions
    models = []
    for i in range(5):
        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(f'model{i}.json')
        models.append(loaded_model)
    return models

def predict_classes(data):
    SEED = 42

    np.random.seed(SEED)
    # aggregated_data = pd.read_csv('your_aggregated_data.csv')
    input_vars = [ 'Xd_max_x', 'Xd_min_x', 'Xd_mean_x',
        'Xd_std_x', 'Xd_rms_x', 'Xd_skew_x', 'Xd_kurtosis_x',
        'Xd_crest_factor_x', 'Xd_form_factor_x', 'Xd_median_x',
        'Xd_variance_x', 'Xd_iqr_x', 'Yd_max_x', 'Yd_min_x', 'Yd_mean_x',
        'Yd_std_x', 'Yd_rms_x', 'Yd_skew_x', 'Yd_kurtosis_x',
        'Yd_crest_factor_x', 'Yd_form_factor_x', 'Yd_median_x',
        'Yd_variance_x', 'Yd_iqr_x', 'Zd_max_x', 'Zd_min_x', 'Zd_mean_x',
        'Zd_std_x', 'Zd_rms_x', 'Zd_skew_x', 'Zd_kurtosis_x',
        'Zd_crest_factor_x', 'Zd_form_factor_x', 'Zd_median_x',
        'Zd_variance_x', 'Zd_iqr_x', 'Xs_max_x', 'Xs_min_x', 'Xs_mean_x',
        'Xs_std_x', 'Xs_rms_x', 'Xs_skew_x', 'Xs_kurtosis_x',
        'Xs_crest_factor_x', 'Xs_form_factor_x', 'Xs_median_x',
        'Xs_variance_x', 'Xs_iqr_x', 'Ys_max_x', 'Ys_min_x', 'Ys_mean_x',
        'Ys_std_x', 'Ys_rms_x', 'Ys_skew_x', 'Ys_kurtosis_x',
        'Ys_crest_factor_x', 'Ys_form_factor_x', 'Ys_median_x',
        'Ys_variance_x', 'Ys_iqr_x', 'Zs_max_x', 'Zs_min_x', 'Zs_mean_x',
        'Zs_std_x', 'Zs_rms_x', 'Zs_skew_x', 'Zs_kurtosis_x',
        'Zs_crest_factor_x', 'Zs_form_factor_x', 'Zs_median_x',
        'Zs_variance_x', 'Zs_iqr_x', 'Xd_max_y', 'Xd_min_y', 'Xd_mean_y', 'Xd_std_y', 'Xd_rms_y',
        'Xd_skew_y', 'Xd_kurtosis_y', 'Xd_crest_factor_y',
        'Xd_form_factor_y', 'Xd_median_y', 'Xd_variance_y', 'Xd_iqr_y',
        'Yd_max_y', 'Yd_min_y', 'Yd_mean_y', 'Yd_std_y', 'Yd_rms_y',
        'Yd_skew_y', 'Yd_kurtosis_y', 'Yd_crest_factor_y',
        'Yd_form_factor_y', 'Yd_median_y', 'Yd_variance_y', 'Yd_iqr_y',
        'Zd_max_y', 'Zd_min_y', 'Zd_mean_y', 'Zd_std_y', 'Zd_rms_y',
        'Zd_skew_y', 'Zd_kurtosis_y', 'Zd_crest_factor_y',
        'Zd_form_factor_y', 'Zd_median_y', 'Zd_variance_y', 'Zd_iqr_y',
        'Xs_max_y', 'Xs_min_y', 'Xs_mean_y', 'Xs_std_y', 'Xs_rms_y',
        'Xs_skew_y', 'Xs_kurtosis_y', 'Xs_crest_factor_y',
        'Xs_form_factor_y', 'Xs_median_y', 'Xs_variance_y', 'Xs_iqr_y',
        'Ys_max_y', 'Ys_min_y', 'Ys_mean_y', 'Ys_std_y', 'Ys_rms_y',
        'Ys_skew_y', 'Ys_kurtosis_y', 'Ys_crest_factor_y',
        'Ys_form_factor_y', 'Ys_median_y', 'Ys_variance_y', 'Ys_iqr_y',
        'Zs_max_y', 'Zs_min_y', 'Zs_mean_y', 'Zs_std_y', 'Zs_rms_y',
        'Zs_skew_y', 'Zs_kurtosis_y', 'Zs_crest_factor_y',
        'Zs_form_factor_y', 'Zs_median_y', 'Zs_variance_y', 'Zs_iqr_y']  # Replace with your input variables
    # outcome_var = 'Failure'  # Replace with your outcome variable
    def ensemble_predict(models, X):
        # Aggregate predictions (probabilities) from each model
        predictions = np.array([model.predict_proba(X) for model in models])
        # Average the probabilities across all models
        avg_predictions = np.mean(predictions, axis=0)
        # Choose the class with the highest average probability
        final_predictions = np.argmax(avg_predictions, axis=1)
        return final_predictions

    X = data[input_vars]
    # Assuming your DataFrame is named sub_sample_df and the column you want to convert is 'categorical_column'
    ensemble_predictions_5 = ensemble_predict(load_model(), X)
    prediction = find_most_frequent_element(ensemble_predictions_5)
    return prediction




def find_most_frequent_element(array):
    # Convert all elements to string to ensure consistency
    array = [str(element) for element in array]
    
    # Use a dictionary to count occurrences of each element
    frequency_count = {}
    for element in array:
        if element in frequency_count:
            frequency_count[element] += 1
        else:
            frequency_count[element] = 1
            
    # Find the element with the highest frequency
    most_frequent = max(frequency_count, key=frequency_count.get)
    return most_frequent

def get_df_vis(df_in,n):
    # Convert 'Time' to datetime if it's not already
    # Function to convert time strings to datetime objects
    df = df_in.copy()
    def convert_to_datetime(time_str):
        time_str = time_str.lstrip()
        return datetime.strptime(time_str, '%H:%M:%S.%f')
    df['Time'] = df['Time'].apply(convert_to_datetime)

    # Step 1: Find the device with the most rows
    device_most_rows = df['Device name'].value_counts().idxmax()

    # Filter the DataFrame for this device
    df_filtered = df[df['Device name'] == device_most_rows]

    # Step 2: Sort by 'Time' to ensure order
    df_filtered = df_filtered.sort_values(by='Time')

    # Check if the DataFrame for the device covers at least n seconds
    if (df_filtered['Time'].max() - df_filtered['Time'].min()).total_seconds() >= n:
        # Calculate the start and end times for the middle n seconds
        time_span = df_filtered['Time'].max() - df_filtered['Time'].min()
        middle_start_time = df_filtered['Time'].min() + (time_span - pd.Timedelta(seconds=n)) / 2
        middle_end_time = middle_start_time + pd.Timedelta(seconds=n)

        # Filter for the middle 5 seconds
        subset_df = df_filtered[(df_filtered['Time'] >= middle_start_time) & (df_filtered['Time'] <= middle_end_time)]
    else:
        # If the time span is less than n seconds, you might return the entire DataFrame or handle as needed
        subset_df = df_filtered 
    return subset_df

def visualize_df_vib(subset_df):
    plt.figure(figsize=(8, 6))
    plt.plot(subset_df['Time'], subset_df['X-axis vibration displacement(um)'], label='X-axis', linestyle='-')
    plt.plot(subset_df['Time'], subset_df['Y-axis vibration displacement(um)'], label='Y-axis', linestyle='--')
    plt.plot(subset_df['Time'], subset_df['Z-axis vibration displacement(um)'], label='Z-axis', linestyle='-.')

    # Adding fancy elements
    plt.title('Vibration Displacement along X, Y, and Z axes', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Vibration Displacement (um)', fontsize=14)
    plt.legend(title='Axis', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Saving the plot as a PNG file

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url
def visualize_df_dis(df):
    plt.figure(figsize=(8, 6))
    plt.plot(df['Time'],df['X-axis vibration speed(mm/s)'], label='X-axis', linestyle='-' )
    plt.plot(df['Time'],df['Y-axis vibration speed(mm/s)'], label='Y-axis', linestyle='--')
    plt.plot(df['Time'],df['Z-axis vibration speed(mm/s)'], label='Z-axis', linestyle='-.')

    # Adding fancy elements
    plt.title('Vibration Speeds along X, Y, and Z axes', fontsize=16)
    plt.xlabel('Measurement Number', fontsize=14)
    plt.ylabel('Vibration Speed (mm/s)', fontsize=14)
    plt.legend(title='Axis', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Saving the plot as a PNG file

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

def encode_image_to_base64(image_path):
    """Encode image to base64."""
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="PNG")  # Assuming PNG format, adjust as necessary
        return base64.b64encode(buffered.getvalue()).decode()

def file_starts_with_number(filename, number):
    """Check if the file starts with the specified number followed by a space."""
    return re.match(r'^{} '.format(number), filename) is not None

def get_images_from_folder(folder_path, start_number):
    """Get the first base64 encoded image from the folder that starts with a specific number."""
    for entry in os.listdir(folder_path):
        if file_starts_with_number(entry, start_number):
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path):
                encoded_image = encode_image_to_base64(full_path)
                return encoded_image  # Return the first matching image
    return None  # Return None if no matching image is found
import re

def extract_description(filename):
    # Remove the numeric prefix and file extension
    description = re.sub(r'^\d+\s*Failure Modes\s*', '', filename)  # Remove numeric prefix and "Failure Modes"
    description = re.sub(r'-\d+(-\d+)*\.png$', '', description)  # Remove trailing numbers and .png extension
    
    return description

import numpy as np
import pandas as pd
# Assuming load_model and other necessary imports are defined

def predictX(data):
    SEED = 42
    np.random.seed(SEED)
    input_vars = [ 'Xd_max_x', 'Xd_min_x', 'Xd_mean_x',
        'Xd_std_x', 'Xd_rms_x', 'Xd_skew_x', 'Xd_kurtosis_x',
        'Xd_crest_factor_x', 'Xd_form_factor_x', 'Xd_median_x',
        'Xd_variance_x', 'Xd_iqr_x', 'Yd_max_x', 'Yd_min_x', 'Yd_mean_x',
        'Yd_std_x', 'Yd_rms_x', 'Yd_skew_x', 'Yd_kurtosis_x',
        'Yd_crest_factor_x', 'Yd_form_factor_x', 'Yd_median_x',
        'Yd_variance_x', 'Yd_iqr_x', 'Zd_max_x', 'Zd_min_x', 'Zd_mean_x',
        'Zd_std_x', 'Zd_rms_x', 'Zd_skew_x', 'Zd_kurtosis_x',
        'Zd_crest_factor_x', 'Zd_form_factor_x', 'Zd_median_x',
        'Zd_variance_x', 'Zd_iqr_x', 'Xs_max_x', 'Xs_min_x', 'Xs_mean_x',
        'Xs_std_x', 'Xs_rms_x', 'Xs_skew_x', 'Xs_kurtosis_x',
        'Xs_crest_factor_x', 'Xs_form_factor_x', 'Xs_median_x',
        'Xs_variance_x', 'Xs_iqr_x', 'Ys_max_x', 'Ys_min_x', 'Ys_mean_x',
        'Ys_std_x', 'Ys_rms_x', 'Ys_skew_x', 'Ys_kurtosis_x',
        'Ys_crest_factor_x', 'Ys_form_factor_x', 'Ys_median_x',
        'Ys_variance_x', 'Ys_iqr_x', 'Zs_max_x', 'Zs_min_x', 'Zs_mean_x',
        'Zs_std_x', 'Zs_rms_x', 'Zs_skew_x', 'Zs_kurtosis_x',
        'Zs_crest_factor_x', 'Zs_form_factor_x', 'Zs_median_x',
        'Zs_variance_x', 'Zs_iqr_x', 'Xd_max_y', 'Xd_min_y', 'Xd_mean_y', 'Xd_std_y', 'Xd_rms_y',
        'Xd_skew_y', 'Xd_kurtosis_y', 'Xd_crest_factor_y',
        'Xd_form_factor_y', 'Xd_median_y', 'Xd_variance_y', 'Xd_iqr_y',
        'Yd_max_y', 'Yd_min_y', 'Yd_mean_y', 'Yd_std_y', 'Yd_rms_y',
        'Yd_skew_y', 'Yd_kurtosis_y', 'Yd_crest_factor_y',
        'Yd_form_factor_y', 'Yd_median_y', 'Yd_variance_y', 'Yd_iqr_y',
        'Zd_max_y', 'Zd_min_y', 'Zd_mean_y', 'Zd_std_y', 'Zd_rms_y',
        'Zd_skew_y', 'Zd_kurtosis_y', 'Zd_crest_factor_y',
        'Zd_form_factor_y', 'Zd_median_y', 'Zd_variance_y', 'Zd_iqr_y',
        'Xs_max_y', 'Xs_min_y', 'Xs_mean_y', 'Xs_std_y', 'Xs_rms_y',
        'Xs_skew_y', 'Xs_kurtosis_y', 'Xs_crest_factor_y',
        'Xs_form_factor_y', 'Xs_median_y', 'Xs_variance_y', 'Xs_iqr_y',
        'Ys_max_y', 'Ys_min_y', 'Ys_mean_y', 'Ys_std_y', 'Ys_rms_y',
        'Ys_skew_y', 'Ys_kurtosis_y', 'Ys_crest_factor_y',
        'Ys_form_factor_y', 'Ys_median_y', 'Ys_variance_y', 'Ys_iqr_y',
        'Zs_max_y', 'Zs_min_y', 'Zs_mean_y', 'Zs_std_y', 'Zs_rms_y',
        'Zs_skew_y', 'Zs_kurtosis_y', 'Zs_crest_factor_y',
        'Zs_form_factor_y', 'Zs_median_y', 'Zs_variance_y', 'Zs_iqr_y'] 

    def ensemble_predict(models, X):
        predictions = np.array([model.predict_proba(X) for model in models])
        avg_predictions = np.mean(predictions, axis=0)
        final_predictions = np.argmax(avg_predictions, axis=1)
        return final_predictions, predictions

    # Load your models here; example:
    models = load_model()

    X = data[input_vars]
    ensemble_predictions, all_predictions = ensemble_predict(models, X)
    prediction = find_most_frequent_element(ensemble_predictions)


    data_point_indices = np.where(ensemble_predictions.reshape(-1,1) == int(prediction))[0].tolist()
    # print("Check", ensemble_predictions[0], prediction)
    # print(ensemble_predictions[0]==prediction)
    print(data_point_indices)
    if len(data_point_indices) > 0:
        # Correctly select one of these data points at random for SHAP analysis
        data_point_index = 0
        selected_data_point = X.iloc[[data_point_index]]  # Double brackets to keep DataFrame format

        # Find the model which gives the highest probability to the majority class for the selected data point
        model_probs = all_predictions[:, data_point_index, int(prediction)]  # Probabilities for majority class by each model
        model_index = np.argmax(model_probs)
        selected_model = models[model_index]  # Assuming 'models' is a list of actual model objects

        print(f"Selected Model Index: {model_index}, Selected Data Point Index: {data_point_index}")

    else:
        print("No data point matches the majority prediction.")
    import shap

    import matplotlib
    matplotlib.use('agg')
    import base64
    from io import BytesIO

    def get_shap_values_plot_path(model, X,pred):
        # Assuming X is your features DataFrame and model is your trained model
        cols = [ 'Xd_max_x', 'Xd_min_x', 'Xd_mean_x',
        'Xd_std_x', 'Xd_rms_x', 'Xd_skew_x', 'Xd_kurtosis_x',
        'Xd_crest_factor_x', 'Xd_form_factor_x', 'Xd_median_x',
        'Xd_variance_x', 'Xd_iqr_x', 'Yd_max_x', 'Yd_min_x', 'Yd_mean_x',
        'Yd_std_x', 'Yd_rms_x', 'Yd_skew_x', 'Yd_kurtosis_x',
        'Yd_crest_factor_x', 'Yd_form_factor_x', 'Yd_median_x',
        'Yd_variance_x', 'Yd_iqr_x', 'Zd_max_x', 'Zd_min_x', 'Zd_mean_x',
        'Zd_std_x', 'Zd_rms_x', 'Zd_skew_x', 'Zd_kurtosis_x',
        'Zd_crest_factor_x', 'Zd_form_factor_x', 'Zd_median_x',
        'Zd_variance_x', 'Zd_iqr_x', 'Xs_max_x', 'Xs_min_x', 'Xs_mean_x',
        'Xs_std_x', 'Xs_rms_x', 'Xs_skew_x', 'Xs_kurtosis_x',
        'Xs_crest_factor_x', 'Xs_form_factor_x', 'Xs_median_x',
        'Xs_variance_x', 'Xs_iqr_x', 'Ys_max_x', 'Ys_min_x', 'Ys_mean_x',
        'Ys_std_x', 'Ys_rms_x', 'Ys_skew_x', 'Ys_kurtosis_x',
        'Ys_crest_factor_x', 'Ys_form_factor_x', 'Ys_median_x',
        'Ys_variance_x', 'Ys_iqr_x', 'Zs_max_x', 'Zs_min_x', 'Zs_mean_x',
        'Zs_std_x', 'Zs_rms_x', 'Zs_skew_x', 'Zs_kurtosis_x',
        'Zs_crest_factor_x', 'Zs_form_factor_x', 'Zs_median_x',
        'Zs_variance_x', 'Zs_iqr_x', 'Xd_max_y', 'Xd_min_y', 'Xd_mean_y', 'Xd_std_y', 'Xd_rms_y',
        'Xd_skew_y', 'Xd_kurtosis_y', 'Xd_crest_factor_y',
        'Xd_form_factor_y', 'Xd_median_y', 'Xd_variance_y', 'Xd_iqr_y',
        'Yd_max_y', 'Yd_min_y', 'Yd_mean_y', 'Yd_std_y', 'Yd_rms_y',
        'Yd_skew_y', 'Yd_kurtosis_y', 'Yd_crest_factor_y',
        'Yd_form_factor_y', 'Yd_median_y', 'Yd_variance_y', 'Yd_iqr_y',
        'Zd_max_y', 'Zd_min_y', 'Zd_mean_y', 'Zd_std_y', 'Zd_rms_y',
        'Zd_skew_y', 'Zd_kurtosis_y', 'Zd_crest_factor_y',
        'Zd_form_factor_y', 'Zd_median_y', 'Zd_variance_y', 'Zd_iqr_y',
        'Xs_max_y', 'Xs_min_y', 'Xs_mean_y', 'Xs_std_y', 'Xs_rms_y',
        'Xs_skew_y', 'Xs_kurtosis_y', 'Xs_crest_factor_y',
        'Xs_form_factor_y', 'Xs_median_y', 'Xs_variance_y', 'Xs_iqr_y',
        'Ys_max_y', 'Ys_min_y', 'Ys_mean_y', 'Ys_std_y', 'Ys_rms_y',
        'Ys_skew_y', 'Ys_kurtosis_y', 'Ys_crest_factor_y',
        'Ys_form_factor_y', 'Ys_median_y', 'Ys_variance_y', 'Ys_iqr_y',
        'Zs_max_y', 'Zs_min_y', 'Zs_mean_y', 'Zs_std_y', 'Zs_rms_y',
        'Zs_skew_y', 'Zs_kurtosis_y', 'Zs_crest_factor_y',
        'Zs_form_factor_y', 'Zs_median_y', 'Zs_variance_y', 'Zs_iqr_y'] 
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Generate SHAP values plot for a specific instance
        matplotlib.pyplot.figure(figsize=(10, 8))
        # shap.summary_plot(shap_values, X, plot_type="bar")
        shap.decision_plot(explainer.expected_value[int(pred)], shap_values[int(pred)][0,:], cols)
        plt.tight_layout()
        # Generate a unique filename using the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f'./static/shap_values_plot_{timestamp}.png'
        p = f'shap_values_plot_{timestamp}.png'
        # Save the plot directly to the specified path
        matplotlib.pyplot.savefig(image_path)
        matplotlib.pyplot.close()
        
        return p
    plot_shap = get_shap_values_plot_path(selected_model,selected_data_point,prediction)
    return prediction, plot_shap

