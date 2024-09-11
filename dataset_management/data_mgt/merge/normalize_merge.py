# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 07:37:01 2024

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

#sys.path.append('../../../ml')
import common

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
   
    
    path = os.path.join(args.datadir, appliance)
    save_path = os.path.join(args.savedir, appliance)

    logger.log(f'Target appliance: {appliance}')

    path = os.path.join(args.datadir, appliance)

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

    max_on_power = common.params_appliance[appliance]['max_on_power']

    # Standardize (or normalize) each dataset and add status.
    for _, file_name in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        save_file_path = os.path.join(save_path, file_name)
        plot = False
        saveit = True
        
        df = load(file_path)

        logger.log(f'\n*** Working on {file_name} ***')
        logger.log('Raw dataset statistics:')
        logger.log(" Size of data set is {:.4f} M rows (count in millions).".format(len(df) / 10 ** 6))
        logger.log(df.loc[:, 'aggregate'].describe())
        logger.log(df.loc[:, appliance].describe())

        
        df_on = df[df['status']==1]
        num_on = len(df_on)
        df_off = df[df['status']==0]
        num_off = len(df_off)
        num_total = df.iloc[:, 2].size
        logger.log(df_on)
        logger.log(f'Number of samples with on status: {num_on}.')
        logger.log(df_on.loc[:, appliance].describe())
        logger.log(df_off)
        logger.log(f'Number of samples with off status: {num_off}.')
        logger.log(f'Number of total samples: {num_total}.')
        assert num_on + num_off == df.iloc[:, 2].size
        
        
        # Check for NaNs.
        print(f'NaNs present: {df.isnull().values.any()}')
        #del df_balanced

        # Standardize aggregate dataset.
        agg_mean = common.ALT_AGGREGATE_MEAN if common.USE_ALT_STANDARDIZATION else train_agg_mean
        agg_std = common.ALT_AGGREGATE_STD if common.USE_ALT_STANDARDIZATION else train_agg_std
        logger.log(f'Standardizing aggregate dataset with mean = {agg_mean} and std = {agg_std}.')
        df.loc[:, 'aggregate'] = (df.loc[:, 'aggregate'] - agg_mean) / agg_std

        # Scale appliance dataset.
        if common.USE_APPLIANCE_NORMALIZATION:
            # Normalize appliance dataset to [0, max_on_power].
            min = 0
            max = max_on_power
            logger.log(f'Normalizing appliance dataset with min = {min} and max = {max}.')
            df.loc[:, appliance] = (df.loc[:, appliance] - min) / (max - min)
        else:
            # Standardize appliance dataset.
            alt_app_mean = common.params_appliance[appliance]['alt_app_mean']
            alt_app_std = common.params_appliance[appliance]['alt_app_std']
            app_mean = alt_app_mean if common.USE_ALT_STANDARDIZATION else train_app_mean
            app_std = alt_app_std if common.USE_ALT_STANDARDIZATION else train_app_std
            logger.log('Using alt standardization.' if common.USE_ALT_STANDARDIZATION 
                  else 'Using default standardization.')
            logger.log(f'Standardizing appliance dataset with mean = {app_mean} and std = {app_std}.')
            df.loc[:, appliance] = (df.loc[:, appliance] - app_mean) / app_std

        ### Other ways of scaling the datasets are commented out below ###
        ### The current method seems to give the best results ###

        # Remove outliers.
        # compute z-scores for all values
        # THIS TAKES FOREVER - DO NOT USE
        #df['z-score'] = df[appliance].apply(lambda x: get_zscore(x, df[appliance]))
        #outliers = df[df['z-score'] > 6]
        #logger.log(outliers)
        #exit()
        
        #logger.log(f'\nStatistics for {file_name} without outliers:')
        #logger.log(df.iloc[:,0].describe())
        #logger.log(df.iloc[:,1].describe())

        # Standardize datasets with training parameters.
        #logger.log(f'\nUsing aggregate training statistics for both datasets.')
        #df.iloc[:,0] = (df.iloc[:,0] - agg_mean) / agg_std
        #df.iloc[:,1] = (df.iloc[:,1] - agg_mean) / agg_std

        # Standardize datasets with respective training parameters.
        #logger.log(f'\nUsing respective training statistics for datasets.')
        #df.iloc[:,0] = (df.iloc[:,0] - agg_mean) / agg_std
        #df.iloc[:,1] = (df.iloc[:,1] - app_mean) / app_std

        # Standardize aggregate dataset.
        #logger.log('\nStandardizing aggregate dataset with training parameters.')
        #df.iloc[:,0] = (df.iloc[:,0] - agg_mean) / agg_std

        # Standardize appliance dataset.
        #logger.log('\nStandardizing appliance dataset with its mean and active train std from NILMTK.')
        #df.iloc[:,1] = (df.iloc[:,1] - df.iloc[:,1].mean()) / 2.140622e+01
        #df.iloc[:,1] = df.iloc[:,1].clip(lower=0.0)

        # Normalize appliance dataset using training parameters.
        #logger.log(f'\nNormalizing appliance dataset with min = {min} and max = {max}.')
        #df.iloc[:,1] = (df.iloc[:,1] - min) / (max - min)

        # Normalize datasets with average training parameters.
        #logger.log(f'\nUsing normalization params mean: {str(545.0)}, std: {str(820.0)}')
        #df.iloc[:,0] = (df.iloc[:,0] - 545.0) / 820.0
        #df.iloc[:,1] = (df.iloc[:,1] - 545.0) / 820.0

        # Normalize appliance dataset to [0, 1].
        #min = df.iloc[:,1].min()
        #max = df.iloc[:,1].max()
        #logger.log(f'Normalizing appliance dataset with min = {min} and max = {max}')
        #df.iloc[:, 1] = (df.iloc[:, 1] - min) / (max - min)

        logger.log(f'Statistics for {file_name} after scaling:')
        logger.log(df.loc[:, 'aggregate'].describe())
        logger.log(df.loc[:, appliance].describe())

        # Check for NaNs.
        logger.log(f'NaNs present: {df.isnull().values.any()}')
        
        filename = os.path.splitext(file_name)[0]
        
        
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
        default_dataset_dir = 'C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/merge/raw/'
        save_directory = 'C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/merge/normalize/'
    
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
                args.savedir,'outputs_log', appliance_name, f'{appliance_name}_normalize.log'
            )
        )
        
        logger.log('Processing started.')
        logger.log(f'Machine name: {socket.gethostname()}')
        logger.log(args)
        main()
        