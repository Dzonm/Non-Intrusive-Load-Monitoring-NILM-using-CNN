# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:33:23 2024

@author: ID
"""


import os
import argparse
import socket
import sys
from logger import Logger

import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import common
#from imblearn.over_sampling import SMOTE


#sys.path.append('../../../ml')
from refit_parameters import params_appliance

SAMPLE_PERIOD = 8

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


def main():
    
    start_time = time.time()

    logger.log(f'Target appliance: {appliance}')

    path = os.path.join(args.datadir, appliance)
    save_path = os.path.join(args.savedir, appliance)

    # Get statistics from training dataset.
    train_file_name = os.path.join(path, f'{appliance}_training_.csv')
    try:
        df = load(train_file_name)
        aggregate_power = df.loc[:, 'aggregate']
        appliance_power = df.loc[:, appliance]

        train_agg_mean = aggregate_power.mean()
        train_agg_std = aggregate_power.std()
        logger.log(f'Training aggregate mean = {train_agg_mean}, std = {train_agg_std}')

        train_app_mean = appliance_power.mean()
        train_app_std = appliance_power.std()
        logger.log(f'Training appliance mean = {train_app_mean}, std = {train_app_std}')

        train_app_min = appliance_power.min()
        train_app_max = appliance_power.max()
        logger.log(f'Training appliance min = {train_app_min}, max = {train_app_max}')

        del df
    except Exception as e:
        sys.exit(e)

    max_on_power = params_appliance[appliance]['max_on_power']

    # Standardize (or normalize) each dataset and add status.
    for _, file_name in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        save_file_path = os.path.join(save_path, file_name)
        plot = True
        saveit = True
        
        df = load(file_path)

        logger.log(f'\n*** Working on {file_name} ***')
        logger.log('Raw dataset statistics:')
        logger.log(" Size of data set is {:.4f} M rows (count in millions).".format(len(df) / 10 ** 6))
        logger.log(df.loc[:, 'aggregate'].describe())
        logger.log(df.loc[:, appliance].describe())
        
        

        # Limit appliance power to [0, max_on_power].
        logger.log(f'Limiting appliance power to [0, {max_on_power}]')
        
        df.loc[:, appliance_name] = df.loc[:, appliance_name].clip(0, max_on_power)
        
        # Get appliance status and add to end of dataframe.
        logger.log('Computing on-off status.')
        status = common.compute_status(df.loc[:, appliance_name].to_numpy(), appliance_name)
        df.insert(2, 'status', status)
        
        df_on = df[df['status']==1]
        num_on = len(df_on)
        df_off = df[df['status']==0]
        num_off = len(df_off)
        num_total = df.iloc[:, 2].size
        #logger.log(df_on)
        logger.log(f'Number of samples with on status: {num_on}.')
        #logger.log(df_on.loc[:, appliance].describe())
        #logger.log(df_off)
        logger.log(f'Number of samples with off status: {num_off}.')
        logger.log(f'Number of total samples: {num_total}.')
        assert num_on + num_off == df.iloc[:, 2].size
        filename = os.path.splitext(file_name)[0]
        
        #Filter out anomalies where aggregate power is less than appliance power.
        #df = df[df['aggregate'] >= df[appliance_name]]
        
        # if 'train' in file_name:
            
        #     # Display the initial distribution of the 'status' column
        #     logger.log("Initial class distribution:")
        #     logger.log(df['status'].value_counts())
            
        #     # Separate samples with status=1 (on) and status=0 (off)
        #     df_on = df[df['status'] == 1]
        #     df_off = df[df['status'] == 0]
            
        #     if len(df_off) > len(df_on):
                
        #         d_value = (1-(((len(df_off)-len(df_on))/len(df))))*len(df_off)
                
        #         # Downsample the 'off' samples to match the number of 'on' samples
        #         df_off_downsampled = df_off.sample(n=int(d_value), random_state=42)
                
        #         # Combine the 'on' samples with the downsampled 'off' samples
        #         df_balanced = pd.concat([df_on, df_off_downsampled])
                
        #         # Sort the DataFrame by index to maintain the original time order
        #         df_balanced = df_balanced.sort_index()
                
        #         # Calculate and log the number of samples for 'on' and 'off' statuses
        #         num_balanced_total = len(df_balanced)
        #         num_balanced_on = len(df_balanced[df_balanced['status'] == 1])
        #         df_on_balanced = df_balanced[df_balanced['status'] == 1]
        #         num_balanced_off = len(df_balanced[df_balanced['status'] == 0])
        #         #df_off_balanced = df_balanced[df_balanced['status'] == 0]
                
        #         # Log the final class distribution after downsampling
        #         logger.log("Class distribution after downsampling:")
        #         logger.log(df_balanced['status'].value_counts())
                
        #         # Display statistical summaries after down-sampling
        #         logger.log('After resampling:')
        #         logger.log("Aggregate description:")
        #         logger.log(df_balanced['aggregate'].describe())
                
        #         logger.log("Appliance description:")
        #         logger.log(df_balanced[appliance].describe())
                
        #         # Aggregate power statistics
        #         aggregate_power = df_balanced['aggregate']
        #         agg_mean = aggregate_power.mean()
        #         agg_std = aggregate_power.std()
        #         logger.log(f'{file_name} aggregate mean = {agg_mean}, std = {agg_std}')
                
        #         # Appliance power statistics
        #         appliance_power = df_balanced[appliance]
        #         app_mean = appliance_power.mean()
        #         app_std = appliance_power.std()
        #         app_min = appliance_power.min()
        #         app_max = appliance_power.max()
        #         logger.log(f'{file_name} appliance mean = {app_mean}, std = {app_std}')
        #         logger.log(f'{file_name} appliance min = {app_min}, max = {app_max}')
                
        #         logger.log(' On status appliance min and max')
        #         on_appliance_min = df_on_balanced[appliance].min()
        #         on_appliance_max = df_on_balanced[appliance].max()
        #         logger.log(f'On appliance min = {on_appliance_min}, max = {on_appliance_max}')
        #         logger.log(f'Number of samples with on status: {num_balanced_on}.')
        #         logger.log(f'Number of samples with off status: {num_balanced_off}.')
        #         logger.log(f'Number of total samples: {num_balanced_total}.')
                
        #         # Log the off-balanced DataFrame for review
        #         #logger.log(df_off_balanced)
                
        #         # Ensure that the sum of 'on' and 'off' samples equals the total number of samples
        #         assert num_balanced_on + num_balanced_off == df_balanced.shape[0]
                
        #         # Extract features (X) and target (y)
        #         X = df_balanced.drop(columns=['status'])
        #         y = df_balanced['status']
                
        #         # Apply SMOTE to balance the dataset
        #         smote = SMOTE(random_state=42)
        #         X_smote, y_smote = smote.fit_resample(X, y)
                
        #         # Combine X_smote and y_smote back into a DataFrame
        #         df_smote_balanced = pd.concat([pd.DataFrame(X_smote, columns=X.columns), pd.DataFrame(y_smote, columns=['status'])], axis=1)
                
        #         # Sort the DataFrame by index to maintain the original time order
        #         df_smote_balanced.sort_index(inplace=True)
                
        #         # Calculate and log the number of samples for 'on' and 'off' statuses after SMOTE
        #         num_smote_total = len(df_smote_balanced)
        #         num_smote_on = len(df_smote_balanced[df_smote_balanced['status'] == 1])
        #         num_smote_off = len(df_smote_balanced[df_smote_balanced['status'] == 0])
                
        #         logger.log(f"Total balanced samples after SMOTE: {num_smote_total}")
        #         logger.log(f"Balanced 'on' samples after SMOTE: {num_smote_on}")
        #         logger.log(f"Balanced 'off' samples after SMOTE: {num_smote_off}")
                
        #         # Ensure that the sum of 'on' and 'off' samples equals the total number of samples
        #         assert num_smote_on + num_smote_off == df_smote_balanced.shape[0]
                
        #         # Continue analysis using the balanced DataFrame
        #         df = df_smote_balanced
                
        # # Check for NaNs.
        # logger.log(f'NaNs present: {df.isnull().values.any()}')
        # #del df_balanced

        if plot & ~('train' in file_name):
            logger.log(f'Aggregate, {appliance_name} and status of {filename}')
            
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
            
            plot_filepath = os.path.join(args.savedir,'outputs_log',f'{appliance_name}', f'{plot_name}')
            logger.log(f'Plot directory of {plot_name}: {plot_filepath}')
            if saveit: plt.savefig(fname=plot_filepath) 
            plt.show()
            plt.close()

        logger.log(f'Statistics for {filename}')
        logger.log("aggregate discription")
        logger.log(df.loc[:, 'aggregate'].describe())
        
        logger.log("appliance discription")
        logger.log(df.loc[:, appliance].describe())

        if plot & ~('train' in file_name):
            
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
            
            plot_filepath = os.path.join(args.savedir, 'outputs_log', f'{appliance_name}', f'{plot_name}.png')
            if saveit: plt.savefig(fname=plot_filepath)
            print(f'Plot saved to: {plot_filepath}')
            plt.show()
            plt.close()
            
        # Check for NaNs.
        logger.log(f'NaNs present?: {df.isnull().values.any()}')

        # Save scaled dataset and overwrite existing csv.
        logger.log(f'*** Saving dataset to {file_path}. ***')
        if saveit: df.to_csv(save_file_path, index=False)
        
        if 'train' in file_name: train = df
        if 'test' in file_name: test = df
        if 'validation' in file_name: val = df
        
        del df
        
    logger.log("    Size of total training set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    logger.log("    Size of total testing set is {:.4f} M rows.".format(len(test) / 10 ** 6))
    logger.log("    Size of total validation set is {:.4f} M rows.".format(len(val) / 10 ** 6))
    
    del train, val, test
    
    logger.log("\nPlease find files in: " + path)
    logger.log("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    logger.log('Processing completed.')
    logger.closefile()

    
    
    
if __name__ == '__main__':
    
    choices=['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
    
    for app in choices:
        
        default_appliance = app
        default_dataset_dir = 'C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/refit/raw/'
        save_directory = 'C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/refit/clean/'
    
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
        parser.add_argument('--savedir',
                            type=str,
                            default=save_directory,
                            help='directory location to save dataset')
        parser.add_argument('--crop',
                            type=int,
                            default=None,
                            help='use part of the dataset for calculations')
    
        args = parser.parse_args()
    
        appliance = args.appliance
        appliance_name = args.appliance
        logger = Logger(
            log_file_name = os.path.join(
                args.savedir,'outputs_log', appliance_name, f'{appliance_name}_clean_resample.log'
            )
        )
        
        logger.log('Processing started.')
        logger.log(f'Machine name: {socket.gethostname()}')
        logger.log(args)
        main()
        