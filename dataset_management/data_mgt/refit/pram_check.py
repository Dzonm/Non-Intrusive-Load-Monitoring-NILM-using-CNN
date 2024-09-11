# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:01:31 2024

@author: ID
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:15:30 2024

@author: ID
"""

"""Scale datasets created by create_new_dataset.py and add on-off status.

Copyright (c) 2023 Lindo St. Angel
"""

import os
import argparse
import sys


import pandas as pd
import numpy as np
#from imblearn.over_sampling import SMOTE


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
# Appliance-specific parameters
params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [2, 3, 5, 9, 11, 20],
        'channels': [8, 9, 8, 7, 7, 9],
        'test_house': 20,
        'val_house': [2, 3, 5, 9, 11],  #20%
        'ttrain_house': [2, 3, 5, 9, 11], #80%
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [2, 3, 5, 9, 11, 20],
        'channels': [5, 8, 7, 6, 6, 8],
        'test_house': 20,
        'val_house': [2, 3, 5, 9, 11],  #20%
        'ttrain_house': [2, 3, 5, 9, 11], #80%
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [2, 3, 5, 9, 11, 20],
        'channels': [1, 3 , 1, 1, 1, 1],
        'test_house': 20,
        'val_house': [2, 3, 5, 9, 11],  #20%
        'ttrain_house': [2, 3, 5, 9, 11], #80%
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [2, 3, 5, 9, 11, 20],
        'channels': [3, 5, 4, 4, 4, 5],
        'test_house': 20,
        'val_house': [2, 3, 5, 9, 11],  #20%
        'ttrain_house': [2, 3, 5, 9, 11], #80%
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [2, 3, 5, 9, 11, 20],
        'channels': [2, 6, 3, 3, 3, 4],
        'test_house': 20,
        'val_house': [2, 3, 5, 9, 11],  #20%
        'ttrain_house': [2, 3, 5, 9, 11], #80%
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




class ApplianceDuration:
    def __init__(self, train_data, appliance_name, threshold):
        """
        Initializes the ApplianceDuration class.

        Args:
        train_data (pd.DataFrame): DataFrame with columns ['aggregate', appliance_name].
        appliance_name (str): The name of the appliance column in the DataFrame.
        threshold (float): Power consumption threshold to determine if the appliance is on.
        """
        self.train_data = train_data  # DataFrame with 'aggregate' and appliance columns
        self.appliance_name = appliance_name  # Name of the appliance column
        self.threshold = threshold  # Threshold to determine on/off state

    def _aggregate_metadata_attribute(self, attribute):
        if attribute == 'min_on_duration':
            return self.min_on_duration()
        elif attribute == 'min_off_duration':
            return self.min_off_duration()

    def min_on_duration(self):
        on_durations = []
        current_on_duration = 0
        timestamps = self.train_data.index  # Assuming timestamps are the index
        for i in range(1, len(self.train_data)):
            if self.train_data[self.appliance_name].iloc[i] > self.threshold:
                current_on_duration += (timestamps[i] - timestamps[i-1]).total_seconds()
            else:
                if current_on_duration > 0:
                    on_durations.append(current_on_duration)
                    current_on_duration = 0
        if current_on_duration > 0:
            on_durations.append(current_on_duration)
        return min(on_durations) if on_durations else None

    def min_off_duration(self):
        off_durations = []
        current_off_duration = 0
        timestamps = self.train_data.index  # Assuming timestamps are the index
        for i in range(1, len(self.train_data)):
            if self.train_data[self.appliance_name].iloc[i] <= self.threshold:
                current_off_duration += (timestamps[i] - timestamps[i-1]).total_seconds()
            else:
                if current_off_duration > 0:
                    off_durations.append(current_off_duration)
                    current_off_duration = 0
        if current_off_duration > 0:
            off_durations.append(current_off_duration)
        return min(off_durations) if off_durations else None


if __name__ == '__main__':
    choices=['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
    default_appliance = 'fridge'
    default_dataset_dir = 'C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/refit/raw/'

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

    # add status.
    for _, file_name in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        
        df = load(file_path)

        print(f'\n*** Working on {file_name} ***')
        print('Raw dataset statistics:')
        print(" Size of data set is {:.4f} M rows (count in millions).".format(len(df) / 10 ** 6))
        print(df.loc[:, 'aggregate'].describe())
        print(df.loc[:, appliance].describe())
        
        aggregate_power = df.loc[:, 'aggregate']
        appliance_power = df.loc[:, appliance]
        agg_mean = aggregate_power.mean()
        agg_std = aggregate_power.std()
        print(f'{file_name} aggregate mean = {agg_mean}, std = {agg_std}')

        app_mean = appliance_power.mean()
        app_std = appliance_power.std()
        print(f'{file_name} appliance mean = {app_mean}, std = {app_std}')

        app_min = appliance_power.min()
        app_max = appliance_power.max()
        print(f'{file_name} appliance min = {app_min}, max = {app_max}')
        
        # # Limit appliance power to [0, max_on_power].
        # print(f'Limiting appliance power to [0, {max_on_power}]')
        
        # df.loc[:, appliance_name] = df.loc[:, appliance_name].clip(0, max_on_power)
        
        # # Get appliance status and add to end of dataframe.
        # print('Computing on-off status.')
        # status = compute_status(df.loc[:, appliance_name].to_numpy(), appliance_name)
        # df.insert(2, 'status', status)

        # df_on = df[df['status']==1]
        # num_on = len(df_on)
        # df_off = df[df['status']==0]
        # num_off = len(df_off)
        # num_total = df.iloc[:, 2].size
        # print(df_on)
        # print(f'Number of samples with on status: {num_on}.')
        # print(df_on.loc[:, appliance].describe())
        # dapp_min = df_on.min()
        # dapp_max = df_on.max()
        # print(f'on appliance min = {dapp_min}, max = {dapp_max}')
        # print(df_off)
        # print(f'Number of samples with off status: {num_off}.')
        # print(f'Number of total samples: {num_total}.')
        # assert num_on + num_off == df.iloc[:, 2].size
        
        
 
        on_power_threshold = params_appliance[appliance]['on_power_threshold']
        
        duration_calculator = ApplianceDuration(train_data= df, appliance_name = appliance_name, threshold =  on_power_threshold)
        min_on_duration = duration_calculator.min_on_duration()
        min_off_duration = duration_calculator.min_off_duration()
        
        print("Minimum On Duration:", min_on_duration)
        print("Minimum Off Duration:", min_off_duration)

        
               
        # if 'train' in file_name:
        #     # Display the initial distribution of the 'status' column
        #     print("Initial class distribution:")
        #     print(df['status'].value_counts())
        
        #     # Separate features (X) and target (y)
        #     X = df.drop(columns=['status'])
        #     y = df['status']
        
        #     # Apply SMOTE to balance the dataset
        #     smote = SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=42, n_jobs=-1)
        #     X_resampled, y_resampled = smote.fit_resample(X, y)
        
        #     # Combine resampled features and target into a new DataFrame
        #     df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        #     df_balanced['status'] = y_resampled
        
        #     # Display the distribution after applying SMOTE
        #     print("Class distribution after SMOTE:")
        #     print(df_balanced['status'].value_counts())
        
        #     # Combine the resampled data with the original 'off' samples
        #     df_balanced = pd.concat([df_balanced, df_off])
        
        #     # Sort the DataFrame by index to maintain the original time order
        #     df_balanced = df_balanced.sort_index()
        
        #     # Calculate and print the number of samples for 'on' and 'off' statuses
        #     num_balanced_total = len(df_balanced)
        #     num_balanced_on = len(df_balanced[df_balanced['status'] == 1])
        #     df_on_balanced = df_balanced[df_balanced['status'] == 1]
        #     num_balanced_off = len(df_balanced[df_balanced['status'] == 0])
        #     df_off_balanced = df_balanced[df_balanced['status'] == 0]
        
        #     # Display statistical summaries after SMOTE-ing/upsampling
        #     print('After SMOTE-ing/upsampling:')
        #     print(df_balanced['aggregate'].describe())
        #     print(df_balanced[appliance].describe())
        
        #     # Aggregate power statistics
        #     aggregate_power = df_balanced['aggregate']
        #     agg_mean = aggregate_power.mean()
        #     agg_std = aggregate_power.std()
        #     print(f'{file_name} aggregate mean = {agg_mean}, std = {agg_std}')
        
        #     # Appliance power statistics
        #     appliance_power = df_balanced[appliance]
        #     app_mean = appliance_power.mean()
        #     app_std = appliance_power.std()
        #     app_min = appliance_power.min()
        #     app_max = appliance_power.max()
        #     print(f'{file_name} appliance mean = {app_mean}, std = {app_std}')
        #     print(f'{file_name} appliance min = {app_min}, max = {app_max}')
        
        #     # On status appliance min and max
        #     on_appliance_min = df_on_balanced.min()
        #     on_appliance_max = df_on_balanced.max()
        #     print(f'On appliance min = {on_appliance_min}, max = {on_appliance_max}')
        #     print(f'Number of samples with on status: {num_balanced_on}.')
        #     print(f'Number of samples with off status: {num_balanced_off}.')
        #     print(f'Number of total samples: {num_balanced_total}.')
        
        #     # Print the off-balanced DataFrame for review
        #     print(df_off_balanced)
        
        #     # Ensure that the sum of 'on' and 'off' samples equals the total number of samples
        #     assert num_balanced_on + num_balanced_off == df_balanced.shape[0]
        
        #     # Continue analysis using the balanced DataFrame
        #     df = df_balanced
        # Check for NaNs.
        print(f'NaNs present: {df.isnull().values.any()}')
        del df