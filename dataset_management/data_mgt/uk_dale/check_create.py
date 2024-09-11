# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:02:47 2024

@author: ID
"""

from ukdale_parameters import params_appliance
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
from functions import load_dataframe
import os

import socket


# Constants
app_choices=['kettle']#, 'microwave', 'fridge', 'dishwasher', 'washingmachine']
#app_name = 'washingmachine'
app_choices=['fridge']



def main():

    start_time = time.time()
    sample_seconds = 8
    training_building_percent = 95
    validation_percent = 13
    plot = False
    saveit = False

    train = pd.DataFrame(columns=['aggregate', appliance_name])
    
    

    for h in params_appliance[appliance_name]['houses']:
        print('    ' + args.data_dir + 'house_' + str(h) + '/'
              + 'channel_' +
              str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]) +
              '.dat')

        mains_df = load_dataframe(args.data_dir, h, 1)
        app_df = load_dataframe(args.data_dir,
                                h,
                                params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)],
                                col_names=['time', appliance_name]
                                )
        
        mains_df['time'] = pd.to_numeric(mains_df['time'], errors='coerce')
        mains_df['time'] = pd.to_datetime(mains_df['time'], unit='s')
        mains_df.set_index('time', inplace=True)
        mains_df.columns = ['aggregate']
        #resample = mains_df.resample(str(sample_seconds) + 'S').mean()
        mains_df.reset_index(inplace=True)
        
        print('==========checking for NaN values==============')
        main_nan_count = mains_df.isnull().sum().sum()
        print(f'mains NaN values: {main_nan_count}')
        app_nan_count = app_df.isnull().sum().sum()
        print(f'appliances NaN values: {app_nan_count}')
        print(" Size of total mains set is {:.4f} M rows.".format(len(mains_df) / 10 ** 6))
        print(" Size of total appliance set is {:.4f} M rows.".format(len(app_df) / 10 ** 6))
        
        if plot:
            print(f'plot of Aggregate for house {h}')
            #print("mains_df:")
            #print(mains_df.head())
            
            # Create a figure and axis
            fig, ax = plt.subplots()
            

            
            # Plot the data
            ax.plot(mains_df['time'], mains_df['aggregate'])
            
            # Set the title and labels
            plot_name = f'AGGREGATE PLOT FOR HOUSE {h}'
            ax.set_title(plot_name)
            ax.set_ylabel('power')
            ax.set_xlabel('Timestamp')
            
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=20)
            
            # Set y-axis limits to ensure it starts at zero
            ax.set_ylim(bottom=0)
                    
            plot_filepath = os.path.join(Save_directory, 'outputs_log',f'{appliance_name}','raw', f'{plot_name}')
            print(f'Plot directory of {plot_name}: {plot_filepath}')
            if saveit: plt.savefig(fname=plot_filepath)
            plt.show()
            plt.close()

        # Appliance
        app_df['time'] = pd.to_numeric(app_df['time'], errors='coerce')
        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')

        if plot:
            print(f'plot of {appliance_name} for house {h}')
            #print("app_df:")
            #print(app_df.head())
            
            # Create a figure and axis
            fig, ax = plt.subplots()
            

            
            # Plot the data
            ax.plot(app_df['time'], app_df[appliance_name])
            
            # Set the title and labels
            plot_name = f'{appliance_name} PLOT FOR HOUSE {h}'
            ax.set_title(plot_name)
            ax.set_ylabel('power')
            ax.set_xlabel('Timestamp')
            
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=20)
            
            # Set y-axis limits to ensure it starts at zero
            ax.set_ylim(bottom=0)
            
            plot_filepath = os.path.join(Save_directory,'outputs_log',f'{appliance_name}','raw', f'{plot_name}')
            print(f'Plot directory of {plot_name}: {plot_filepath}')
            if saveit: plt.savefig(fname=plot_filepath)
            plt.show()
            plt.close()

        # the timestamps of mains and appliance are not the same, we need to align them
        # 1. join the aggragte and appliance dataframes;
        # 2. interpolate the missing values by forward fill;
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)
        
        df_align = mains_df.join(app_df, how='outer').resample(str(sample_seconds) + 'S').mean().bfill(limit=1)
        
        
        
        print(f'Data in which the aggregate power is less than the main power for house {h}')
        df_anomaly = df_align[df_align['aggregate'] < df_align[appliance_name]]
        print(df_anomaly)
        anomaly_count = len(df_anomaly)
        print(f'Total anomalies present: {anomaly_count} rows out of {len(df_align) / 10 ** 6}M rows')
        
        
        
        # Check for NaNs.     
        print("check for number of NaNs after combining the aggregate and appliance dataframes")
        print(f'NaNs present ?: {df_align.isnull().values.any()}')
        nan_count = df_align.isnull().sum().sum()
        print(f'Total NaNs present: {nan_count}')
             
        df_align = df_align.dropna()
        
        print("Check for NaNs affer removing NaNs")     
        print(f"check if there are NaNs?:{df_align.isnull().values.any()}")
        nan_count = df_align.isnull().sum().sum()
        print(f'Total NaNs present: {nan_count}')
        print(" Size of after of Data removing NaNs is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
        

        df_align.reset_index(inplace=True)


        del mains_df, app_df, df_align['time']
        
            

        if h == params_appliance[appliance_name]['test_build']:
            # Test CSV
            if plot:
                print(f'Aggregate, {appliance_name} and status PLOT FOR HOUSE {h} used for TESTING')
                
                # Create a figure and axis
                fig, ax = plt.subplots()
                
                # Plot the data
                ax.plot(df_align['aggregate'].values)
                ax.plot(df_align[appliance_name].values)
                #ax.plot((df_align['status'].values)*(df_align[appliance_name].values),linestyle='dotted')
                ax.grid()
                
                # Set the legend, title, and labels
                ax.legend(['Aggregate', appliance_name],loc='upper right')
                plot_name = f'Aggregate and {appliance_name} of house {h} - TESTING'
                ax.set_title(plot_name)
                ax.set_ylabel('power')
                ax.set_xlabel('sample')
                
                # Set y-axis limits to ensure it starts at zero
                ax.set_ylim(bottom=0)
                
                plot_filepath = os.path.join(Save_directory,'outputs_log',f'{appliance_name}','raw', f'{plot_name}')
                print(f'Plot directory of {plot_name}: {plot_filepath}')
                if saveit: plt.savefig(fname=plot_filepath)
                plt.show()
                plt.close()
            
            df_align.to_csv(args.save_path + appliance_name +'_test_.csv',index=False)
            print(" Size of test set is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
            print(f'house{h} dataset: use for testing')
            print(df_align['aggregate'].describe())
            print(df_align[appliance_name].describe())
            
            continue
        
        train = pd.concat([train, df_align], ignore_index=True)
        
        
        del df_align

    
    print(f'house{h} dataset: use for tranning and validation')
    print(" Size of tranning and validation set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    print(train['aggregate'].describe())
    print(train[appliance_name].describe())
    
    
       

    # Validation CSV
    val_len = int((len(train)/100)*validation_percent)
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    train.drop(train.index[-val_len:], inplace=True)
    # Validation CSV
    val.to_csv(args.save_path + appliance_name + '_validation_' + '.csv',index=False)
    
    #house for train & validation
    h = params_appliance[appliance_name]['train_build']
    
    #validation plot
    if plot:
        print(f'Aggregate, {appliance_name} and status house {h} which is used for VALIDATION')
        
        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Plot the data
        ax.plot(val['aggregate'].values)
        ax.plot(val[appliance_name].values)
        #ax.plot((val['status'].values)*(val[appliance_name].values), linestyle='dotted')
        ax.grid()
        
        # Set the legend, title, and labels
        ax.legend(['Aggregate', appliance_name],loc='upper right')
        plot_name = f'Aggregate and {appliance_name} of house {h} - VALIDATION PLOT'
        ax.set_title(plot_name)
        ax.set_ylabel('Power')
        ax.set_xlabel('sample')
        
        # Set y-axis limits to ensure it starts at zero
        ax.set_ylim(bottom=0)
    
        
        
        plot_filepath = os.path.join(Save_directory,'outputs_log',f'{appliance_name}','raw', f'{plot_name}')
        print(f'Plot directory of {plot_name}: {plot_filepath}')
        if saveit: plt.savefig(fname=plot_filepath)
        plt.show()
        plt.close()
        
    # Training CSV
    train.to_csv(args.save_path + appliance_name + '_training_.csv',index=False)
    if plot:
        print(f'Aggregate, {appliance_name} and status of house {h} - TRAINING')
        
        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Plot the data
        ax.plot(train['aggregate'].values)
        ax.plot(train[appliance_name].values)
        #ax.plot((train['status'].values)*(train[appliance_name].values),linestyle='dotted')
        ax.grid()
        
        # Set the legend, title, and labels
        ax.legend(['Aggregate', appliance_name],loc='upper right')
        plot_name = f'Aggregate and {appliance_name} of house {h} - TRAINING PLOT'
        ax.set_title(plot_name)
        ax.set_ylabel('Power')
        ax.set_xlabel('sample')
        
        # Set y-axis limits to ensure it starts at zero
        ax.set_ylim(bottom=0)
        
        plot_filepath = os.path.join(Save_directory,'outputs_log',f'{appliance_name}','raw', f'{plot_name}')
        print(f'Plot directory of {plot_name}: {plot_filepath}')
        if saveit: plt.savefig(fname=plot_filepath)
        plt.show()
        plt.close()
        
    print("    Size of total training set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    print("    Size of total validation set is {:.4f} M rows.".format(len(val) / 10 ** 6))
    del train, val


    print("\nPlease find files in: " + args.save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    print('Processing completed.')
    


if __name__ == '__main__':
    for app in app_choices:
        app_name = app
        DATA_DIRECTORY = 'C:/Users/ID/nilmtk_test/mynilm/UK-DALE/'
        Save_directory = "C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/uk_dale/"
        SAVE_PATH = Save_directory + f'{app_name}/'



        def get_arguments():
            parser = argparse.ArgumentParser(description='sequence to point learning \
                                             example for NILM')
            parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                                  help='The directory containing the UKDALE data')
            parser.add_argument('--appliance_name', type=str, default=app_name,
                                  help='which appliance you want to train: kettle,\
                                  microwave,fridge,dishwasher,washingmachine')
            parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                                  help='The directory to store the training data')
            return parser.parse_args()


        args = get_arguments()
        appliance_name = args.appliance_name
        
        
        print('Processing started.')
        print(f'Machine name: {socket.gethostname()}')
        print(args)
        main()
        
        

        