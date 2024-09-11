

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:33:23 2024

@author: ID
"""


import os
import argparse
import socket
import sys


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import common
from imblearn.over_sampling import SMOTE


#sys.path.append('../../../ml')
#from ukdale_parameters import params_appliance

SAMPLE_PERIOD = 8



#sys.path.append('../../../ml')
# Alternative aggregate standardization parameters used for all appliances.
# From Michele D’Incecco, et. al., "Transfer Learning for Non-Intrusive Load Monitoring"
ALT_AGGREGATE_MEAN = 522.0  # in Watts
ALT_AGGREGATE_STD = 814.0   # in Watts

# If True the alternative standardization parameters will be used
# for scaling the datasets.
USE_ALT_STANDARDIZATION = False

# If True the appliance dataset will be normalized to [0, max_on_power]
# else the appliance dataset will be z-score standardized.
USE_APPLIANCE_NORMALIZATION = True

# Power consumption sample update period in seconds.
SAMPLE_PERIOD = 8

# Various parameters used for training, validation and testing.
# Except where noted, values are calculated from statistical analysis
# of the respective dataset.
# Various parameters used for training, validation and testing.
# Except where noted, values are calculated from statistical analysis
# of the respective dataset.
params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [1, 2],
        'channels': [10, 8],
        'train_build': [1],
        'test_build': 2,
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [1, 2],
        'channels': [13, 15],
        'train_build': [1],
        'test_build': 2,
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [1, 2],
        'channels': [12, 14],
        'train_build': [1],
        'test_build': 2,
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [1, 2],
        'channels': [6, 13],
        'train_build': [1],
        'test_build': 2,
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2],
        'channels': [5, 12],
        'train_build': [1],
        'test_build': 2,
    }
}


def load(file_name, crop=None):
    """Load input dataset file."""
    df = pd.read_csv(file_name, header=0, nrows=crop)

    return df

def compute_stats(df) -> dict:
    """ Given a Series DataFrame compute its statistics. """
    return {
        'mean': df.mean(),
        'std': df.std(),
        'median': df.median(),
        'quartile1': df.quantile(q=0.25, interpolation='lower'),
        'quartile3': df.quantile(q=0.75, interpolation='lower')
    }

def get_zscore(value, values):
    """Obtain the z-score of a given value"""
    m = np.mean(values)
    s = np.std(values)
    z_score = (value - m)/s
    return np.abs(z_score)

def compute_status(appliance_power:np.ndarray, appliance:str) -> list:
    """Compute appliance on-off status."""
    threshold = params_appliance[appliance]['on_power_threshold']

    def ceildiv(a:int, b:int) -> int:
        """Upside-down floor division."""
        return -(a // -b)

    # Convert durations from seconds to samples.
    min_on_duration = ceildiv(common.params_appliance[appliance]['min_on_duration'],
                              SAMPLE_PERIOD)
    min_off_duration = ceildiv(common.params_appliance[appliance]['min_off_duration'],
                               SAMPLE_PERIOD)

    # Apply threshold to appliance powers.
    initial_status = appliance_power.copy() >= threshold

    # Find transistion indices.
    status_diff = np.diff(initial_status)
    events_idx = status_diff.nonzero()
    events_idx = np.array(events_idx).squeeze()
    events_idx += 1

    # Adjustment for first and last transition.
    if initial_status[0]:
        events_idx = np.insert(events_idx, 0, 0)
    if initial_status[-1]:
        events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

    # Separate out on and off events.
    events_idx = events_idx.reshape((-1, 2))
    on_events = events_idx[:, 0].copy()
    off_events = events_idx[:, 1].copy()
    assert len(on_events) == len(off_events)

    # Filter out on and off transitions faster than minimum values.
    if len(on_events) > 0:
        off_duration = on_events[1:] - off_events[:-1]
        off_duration = np.insert(off_duration, 0, 1000)
        on_events = on_events[off_duration > min_off_duration]
        off_events = off_events[np.roll(off_duration, -1) > min_off_duration]

        on_duration = off_events - on_events
        on_events = on_events[on_duration >= min_on_duration]
        off_events = off_events[on_duration >= min_on_duration]
        assert len(on_events) == len(off_events)

    # Generate final status.
    status = [0] * appliance_power.size
    for on, off in zip(on_events, off_events):
        status[on: off] = [1] * (off - on)

    return status

if __name__ == '__main__':
    choices=['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
    default_appliance = 'kettle'
    default_dataset_dir = 'C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/uk_dale/clean/'
    

    parser = argparse.ArgumentParser(
        description='scale a dataset'
    )
    parser.add_argument('--appliance',
                        type=str,
                        default=default_appliance,
                        help='name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default=default_dataset_dir,
                        help='directory location dataset')
    parser.add_argument('--crop',
                        type=int,
                        default=None,
                        help='use part of the dataset for calculations')

    args = parser.parse_args()

    appliance = args.appliance
    appliance_name = args.appliance
    
    print(f'Machine name: {socket.gethostname()}')
    print(args)


    print(f'Target appliance: {appliance}')

    path = os.path.join(args.datadir, appliance)

    # Get statistics from training dataset.
    train_file_name = os.path.join(path, f'{appliance}_training_.csv')
    try:
        df = load(train_file_name)
        aggregate_power = df.loc[:, 'aggregate']
        appliance_power = df.loc[:, appliance]

        train_agg_mean = aggregate_power.mean()
        train_agg_std = aggregate_power.std()
        print(f'Training aggregate mean = {train_agg_mean}, std = {train_agg_std}')

        train_app_mean = appliance_power.mean()
        train_app_std = appliance_power.std()
        print(f'Training appliance mean = {train_app_mean}, std = {train_app_std}')

        train_app_min = appliance_power.min()
        train_app_max = appliance_power.max()
        print(f'Training appliance min = {train_app_min}, max = {train_app_max}')

        del df
    except Exception as e:
        sys.exit(e)

    max_on_power = params_appliance[appliance]['max_on_power']

    # Standardize (or normalize) each dataset and add status.
    for _, file_name in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        plot = True
        saveit = False
        
        df = load(file_path)

        print(f'\n*** Working on {file_name} ***')
        print('Raw dataset statistics:')
        print(" Size of data set is {:.4f} M rows (count in millions).".format(len(df) / 10 ** 6))
        print(df.loc[:, 'aggregate'].describe())
        print(df.loc[:, appliance].describe())
        
        

        # Limit appliance power to [0, max_on_power].
        print(f'Limiting appliance power to [0, {max_on_power}]')
        
        df.loc[:, appliance_name] = df.loc[:, appliance_name].clip(0, max_on_power)
        # Get appliance status and add to end of dataframe.
        print('Computing on-off status.')
        status = compute_status(df.loc[:, appliance_name].to_numpy(), appliance_name)
        df.insert(2, 'status', status)
        
        df_on = df[df['status']==1]
        num_on = len(df_on)
        df_off = df[df['status']==0]
        num_off = len(df_off)
        num_total = df.iloc[:, 2].size
        #print(df_on)
        print(f'Number of samples with on status: {num_on}.')
        #print(df_on.loc[:, appliance].describe())
        #print(df_off)
        print(f'Number of samples with off status: {num_off}.')
        print(f'Number of total samples: {num_total}.')
        assert num_on + num_off == df.iloc[:, 2].size
        filename = os.path.splitext(file_name)[0]
        
        #Filter out anomalies where aggregate power is less than appliance power.
        df = df[df['aggregate'] >= df[appliance_name]]
        
        
        if 'train' in file_name:
            
            # Display the initial distribution of the 'status' column
            print("Initial class distribution:")
            print(df['status'].value_counts())
            
            # Separate samples with status=1 (on) and status=0 (off)
            df_on = df[df['status'] == 1]
            df_off = df[df['status'] == 0]
            
            # Downsample the 'off' samples to match the number of 'on' samples
            df_off_downsampled = df_off.sample(n=int(len(df)*0.4), random_state=42)
            
            # Combine the 'on' samples with the downsampled 'off' samples
            df_balanced = pd.concat([df_on, df_off_downsampled])
            
            # Sort the DataFrame by index to maintain the original time order
            df_balanced = df_balanced.sort_index()
            
            # Calculate and log the number of samples for 'on' and 'off' statuses
            num_balanced_total = len(df_balanced)
            num_balanced_on = len(df_balanced[df_balanced['status'] == 1])
            df_on_balanced = df_balanced[df_balanced['status'] == 1]
            num_balanced_off = len(df_balanced[df_balanced['status'] == 0])
            df_off_balanced = df_balanced[df_balanced['status'] == 0]
            
            # Log the final class distribution after downsampling
            print("Class distribution after downsampling:")
            print(df_balanced['status'].value_counts())
            
            # Display statistical summaries after down-sampling
            print('After resampling:')
            print("Aggregate description:")
            print(df_balanced['aggregate'].describe())
            
            print("Appliance description:")
            print(df_balanced[appliance].describe())
            
            # Aggregate power statistics
            aggregate_power = df_balanced['aggregate']
            agg_mean = aggregate_power.mean()
            agg_std = aggregate_power.std()
            print(f'{file_name} aggregate mean = {agg_mean}, std = {agg_std}')
            
            # Appliance power statistics
            appliance_power = df_balanced[appliance]
            app_mean = appliance_power.mean()
            app_std = appliance_power.std()
            app_min = appliance_power.min()
            app_max = appliance_power.max()
            print(f'{file_name} appliance mean = {app_mean}, std = {app_std}')
            print(f'{file_name} appliance min = {app_min}, max = {app_max}')
            
            # On status appliance min and max
            on_appliance_min = df_on_balanced[appliance].min()
            on_appliance_max = df_on_balanced[appliance].max()
            print(f'On appliance min = {on_appliance_min}, max = {on_appliance_max}')
            print(f'Number of samples with on status: {num_balanced_on}.')
            print(f'Number of samples with off status: {num_balanced_off}.')
            print(f'Number of total samples: {num_balanced_total}.')
            
            # Log the off-balanced DataFrame for review
            print(df_off_balanced)
            
            # Ensure that the sum of 'on' and 'off' samples equals the total number of samples
            assert num_balanced_on + num_balanced_off == df_balanced.shape[0]
            
            # Extract features (X) and target (y)
            X = df_balanced.drop(columns=['status'])
            y = df_balanced['status']
            
            # Apply SMOTE to balance the dataset
            smote = SMOTE(random_state=42)
            X_smote, y_smote = smote.fit_resample(X, y)
            
            # Combine X_smote and y_smote back into a DataFrame
            df_smote_balanced = pd.concat([pd.DataFrame(X_smote, columns=X.columns), pd.DataFrame(y_smote, columns=['status'])], axis=1)
            
            # Sort the DataFrame by index to maintain the original time order
            #df_smote_balanced.sort_index(inplace=True)
            
            # Calculate and log the number of samples for 'on' and 'off' statuses after SMOTE
            num_smote_total = len(df_smote_balanced)
            num_smote_on = len(df_smote_balanced[df_smote_balanced['status'] == 1])
            num_smote_off = len(df_smote_balanced[df_smote_balanced['status'] == 0])
            
            print(f"Total balanced samples after SMOTE: {num_smote_total}")
            print(f"Balanced 'on' samples after SMOTE: {num_smote_on}")
            print(f"Balanced 'off' samples after SMOTE: {num_smote_off}")
            
            # Ensure that the sum of 'on' and 'off' samples equals the total number of samples
            assert num_smote_on + num_smote_off == df_smote_balanced.shape[0]
            
            # Continue analysis using the balanced DataFrame
            df = df_smote_balanced
        # Check for NaNs.
        print(f'NaNs present: {df.isnull().values.any()}')
        #del df_balanced

        # Check for NaNs.
        print(f'NaNs present: {df.isnull().values.any()}')
        #del df_balanced

        if plot:
            print(f'Aggregate, {appliance_name} and status of {filename}')
            
            # Create a figure and axis
            fig, ax = plt.subplots()
            
            # Plot the data
            ax.plot(df['aggregate'].values)
            ax.plot(df[appliance_name].values)
            ax.plot((df['status'].values)*(df[appliance_name].values)*0.7,linestyle='dotted')
            ax.grid()
            
            # Set the legend, title, and labels
            ax.legend(['Aggregate', appliance_name,'on status'],loc='upper right')
            plot_name = f'Aggregate, {appliance_name} and status of {filename}'
            ax.set_title(plot_name)
            ax.set_ylabel('Power')
            ax.set_xlabel('sample')
            
            # Set y-axis limits to ensure it starts at zero
            ax.set_ylim(bottom=0)
            
            plot_filepath = os.path.join(args.datadir,'outputs_log',f'{appliance_name}','clean', f'{plot_name}')
            print(f'Plot directory of {plot_name}: {plot_filepath}')
            if saveit: plt.savefig(fname=plot_filepath) 
            plt.show()
            plt.close()

        print(f'Statistics for {filename} after cleaning and resampling:')
        print("aggregate discription")
        print(df.loc[:, 'aggregate'].describe())
        
        print("appliance discription")
        print(df.loc[:, appliance].describe())

        if plot:
            
            # Show dataset histograms.
            
            # Create a figure with two subplots, one above the other
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
            
            if 'train' in file_name:
                plot_tt_agg = f'Cleansed and Resampled Histogram for {filename}: Aggregate'
                plot_tt_app = f'Cleansed and Resampled Histogram for {filename}: {appliance}'
                plot_name = f'Cleansed and Resampled Histogram for {filename}'
            else:
                plot_tt_agg = f'Cleansed Histogram for {filename}: Aggregate'
                plot_tt_app = f'Cleansed Histogram for {filename}: {appliance}'
                plot_name = f'Cleansed Histogram for {filename}'
            # Plot the histogram for 'aggregate' on the first subplot (ax1)
            df.loc[:, 'aggregate'].hist(ax=ax1)
            
            ax1.set_title(plot_tt_agg)
            ax1.set_xlabel('Power Consumption (W)')
            ax1.set_ylabel('Frequency')
            
            # Plot the histogram for the appliance data on the second subplot (ax2)
            df.loc[:, appliance].hist(ax=ax2)
            
            ax2.set_title(plot_tt_app)
            ax2.set_xlabel('Power Consumption (W)')
            ax2.set_ylabel('Frequency')
            
            # Save and show the plot
            
            plot_filepath = os.path.join(args.datadir, 'outputs_log', f'{appliance_name}', 'clean', f'{plot_name}.png')
            if saveit: plt.savefig(fname=plot_filepath)
            print(f'Plot saved to: {plot_filepath}')
            plt.show()
            plt.close()
            
        # Check for NaNs.
        print(f'NaNs present?: {df.isnull().values.any()}')

        # Save scaled dataset and overwrite existing csv.
        print(f'*** Saving dataset to {file_path}. ***')
        if saveit: df.to_csv(file_path, index=False)

        if 'train' in file_name: train = df
        if 'test' in file_name: test = df
        if 'validation' in file_name: val = df
        
        del df
        
    print("    Size of total training set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    print("    Size of total testing set is {:.4f} M rows.".format(len(test) / 10 ** 6))
    print("    Size of total validation set is {:.4f} M rows.".format(len(val) / 10 ** 6))
    
    del train, val, test