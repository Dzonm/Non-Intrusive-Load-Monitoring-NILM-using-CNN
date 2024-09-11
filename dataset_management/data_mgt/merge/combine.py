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
import common
import matplotlib.pyplot as plt



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

def main():
    
    start_time = time.time()

    logger.log(f'Target appliance: {appliance}')

    path2 = os.path.join(args.datadir,'uk_dale','clean', appliance)
    path1 = os.path.join(args.datadir,'refit','clean', appliance)
    df_data = pd.DataFrame(columns=['aggregate', appliance_name])
    
    save_path = os.path.join(args.savedir, appliance)

    max_on_power = common.params_appliance[appliance]['max_on_power']
    
    for _, file_name in enumerate(os.listdir(path1)):
        file_path1 = os.path.join(path1, file_name)
        file_path2 = os.path.join(path2, file_name)
        save_file_path = os.path.join(save_path, file_name)
        plot = False
        saveit = True
        
        df1 = load(file_path1)
        
        df =pd.concat([df_data, df1], ignore_index=True)
        
        df2 = load(file_path2)
        
        df =pd.concat([df, df2], ignore_index=True)

        logger.log(f'\n*** Working on {file_name} ***')
        logger.log('merged dataset statistics:')
        logger.log(" Description:")
        
        logger.log(" Size of data set is {:.4f} M rows (count in millions).".format(len(df) / 10 ** 6))
        # Limit appliance power to [0, max_on_power].
        logger.log(f'Limiting appliance power to [0, {max_on_power}]')
        
        df.loc[:, appliance_name] = df.loc[:, appliance_name].clip(0, max_on_power)
        
        
        logger.log(df.loc[:, 'aggregate'].describe())
        logger.log(df.loc[:, appliance].describe())
     
        df_on = df[df['status']==1]
        num_on = len(df_on)
        df_off = df[df['status']==0]
        num_off = len(df_off)
        num_total = df.iloc[:, 2].size
        #logger.log(df_on)\\
            
        
        filename = os.path.splitext(file_name)[0]
     
        # Aggregate power statistics
        aggregate_power = df['aggregate']
        agg_mean = aggregate_power.mean()
        agg_std = aggregate_power.std()
        logger.log(f'{filename} aggregate mean = {agg_mean}, std = {agg_std}')
        
        # Appliance power statistics
        appliance_power = df[appliance]
        app_mean = appliance_power.mean()
        app_std = appliance_power.std()
        app_min = appliance_power.min()
        app_max = appliance_power.max()
        logger.log(f'{filename} appliance mean = {app_mean}, std = {app_std}')
        logger.log(f'{filename} appliance min = {app_min}, max = {app_max}')
        
        logger.log(' On status appliance min and max')
        on_appliance_min = df_on[appliance].min()
        on_appliance_max = df_on[appliance].max()
        logger.log(f'On appliance min = {on_appliance_min}, max = {on_appliance_max}')
        logger.log(f'Number of samples with on status: {num_on}.')
        logger.log(f'Number of samples with off status: {num_off}.')
        logger.log(f'Number of total samples: {num_total}.')
        assert num_on + num_off == df.iloc[:, 2].size        
        # Check for NaNs.
        logger.log(f'NaNs present: {df.isnull().values.any()}')
        #del df_balanced

        if plot:
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

        # logger.log(f'Statistics for {filename}')
        # logger.log("aggregate discription")
        # logger.log(df.loc[:, 'aggregate'].describe())
        
        # logger.log("appliance discription")
        # logger.log(df.loc[:, appliance].describe())

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

       
        logger.log(f'*** Saving dataset to {save_file_path}. ***')
        if saveit: df.to_csv(save_file_path, index=False)
        
        if 'train' in file_name: train = df
        if 'test' in file_name: test = df
        if 'validation' in file_name: val = df
        
        del df
        
    logger.log("    Size of total training set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    logger.log("    Size of total testing set is {:.4f} M rows.".format(len(test) / 10 ** 6))
    logger.log("    Size of total validation set is {:.4f} M rows.".format(len(val) / 10 ** 6))
    
    del train, val, test
    
    logger.log("\nPlease find files in: " + save_path)
    logger.log("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    logger.log('Processing completed.')
    logger.closefile()

    
    
    
if __name__ == '__main__':
    
    choices=['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
    
    for app in choices:
        
        default_appliance = app
        default_dataset_dir = 'C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/'
        save_directory = 'C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/merge/raw/'
    
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
                args.savedir,'outputs_log', appliance_name, f'{appliance_name}_merge_.log'
            )
        )
        
        logger.log('Processing started.')
        logger.log(f'Machine name: {socket.gethostname()}')
        logger.log(args)
        main()
        