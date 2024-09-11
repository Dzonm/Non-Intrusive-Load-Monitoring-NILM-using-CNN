# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 23:31:56 2024

@author: ID
"""



import time
import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from logger import Logger
import socket

import common

# Constants
app_choices=['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
#app_name = 'washingmachine'

saveit = True
plot =True
training_building_percent = 80
validation_percent = 20
sample_seconds = 8



# Appliance-specific parameters
params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3000,
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
        'on_power_threshold': 10,
        'max_on_power': 1500,
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
        'on_power_threshold': 20,
        'max_on_power': 3000,
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
        'on_power_threshold': 30,
        'max_on_power': 2600,
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
        'max_on_power': 2700,
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


def load_data(path, building, appliance, channel):
    file_name = os.path.join(path, f'CLEAN_House{building}.csv')
    return pd.read_csv(file_name, header=0, names=['time', 'aggregate', appliance], usecols=[1, 2, channel+2], parse_dates=True, memory_map=True)

def compute_stats(df):
    return {
        'mean': df.mean(),
        'std': df.std(),
        'median': df.median(),
        'quartile1': df.quantile(q=0.25),
        'quartile3': df.quantile(q=0.75)
    }

def save_plot(data, appliance_name, plot_name, save_dir,logger):
    if plot:
        fig, ax = plt.subplots()
        ax.plot(data['aggregate'].values)
        ax.plot(data[appliance_name].values)
        ax.grid()
        ax.legend(['Aggregate', appliance_name], loc='upper right')
        ax.set_title(plot_name)
        ax.set_ylabel('Power')
        ax.set_xlabel('Sample')
        ax.set_ylim(bottom=0)
        logger.log(f'Processing plot for : {appliance_name}')
        plot_filepath = os.path.join(save_dir,'outputs_log', appliance_name, f'{plot_name}.png')
        os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
        if saveit: 
            plt.savefig(plot_filepath)
            logger.log(f'saved plot for : {appliance_name} to {plot_filepath}')
        plt.show()
        plt.close()
    
    
def discib(data, house_number, appliance_name, logger):
    logger.log("Checking for number of NaNs in the dataframes")
    logger.log(f'NaNs present ?: {data.isnull().values.any()}')
    nan_count = data.isnull().sum().sum()
    logger.log(f'Total NaNs present: {nan_count}')
    
    logger.log(f'Data in which the aggregate power is less than the main power for house {house_number}')
    print(data)
    df_anomaly = data[data['aggregate'] < data[appliance_name]]
    df_anomaly.reset_index(inplace=True)
    logger.log(df_anomaly)
    
    anomaly_count = len(df_anomaly)
    max_on_power = params_appliance[appliance_name]['max_on_power']
    
    logger.log(f'Anomalies present for house {house_number}: {anomaly_count} rows out of {len(data) / 10 ** 6}M rows')
    # Limit appliance power to [0, max_on_power].
    
    
    df = pd.DataFrame(columns=['aggregate', appliance_name])
    df = pd.concat([df, data], ignore_index=True)
    
    #logger.log(f'Limiting appliance power to [0, {max_on_power}]')
    df.loc[:, appliance_name] = df.loc[:, appliance_name].clip(0, max_on_power)
    # Get appliance status and add to end of dataframe.
    
    logger.log('Computing on-off status.')
    status = common.compute_status(df.loc[:, appliance_name].to_numpy(), appliance_name)
    
  
    df.insert(2, 'status', status)
    
    df_on = df[df['status'] == 1]
    num_on = len(df_on)
    df_off = df[df['status'] == 0]
    num_off = len(df_off)
    num_total = df.iloc[:, 2].size
    logger.log(f'Number of samples with on status: {num_on}.')
    logger.log(f'Number of samples with off status: {num_off}.')
    logger.log(f'Number of total samples: {num_total}.')
    del df,anomaly_count,df_off,df_on,num_off,num_on,num_total



def main():
    start_time = time.time()
    args = get_arguments()
    
    appliance = args.appliance_name
    appliance_name = args.appliance_name
    logger = Logger(
        log_file_name=os.path.join(
            args.save_path, 'outputs_log', appliance_name, f'{appliance_name}_create.log'
        )
    )
    
    logger.log('start pre-processing')
    logger.log(f'Machine name: {socket.gethostname()}')
    logger.log(args)

    logger.log(f'Target appliance: {appliance}')
    
    appliance_name = args.appliance_name
    path = args.data_dir
    save_path = os.path.join(args.save_path, appliance_name)
    os.makedirs(save_path, exist_ok=True)
    
    logger.log(f'Processing appliance: {appliance_name}')
    logger.log(f'Data directory: {path}')
    logger.log(f'Save directory: {save_path}')
    
    total_length = 0
    numbers = []
    
    train = pd.DataFrame(columns=['aggregate', appliance_name])
    
    for filename in os.listdir(path):
        if "CLEAN_House" in filename:
            #try:
            house_number = int(re.search(r'\d+', filename).group())
            
            if house_number in params_appliance[appliance_name]['houses']:
                
                chnl = params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(house_number)]
                data = load_data(path, house_number, appliance_name, chnl)
                
                logger.log(f'Processing file: {filename}, {appliance_name} - channel {chnl}')
                logger.log("Checking for number of NaNs in the dataframes before date-time formating")
                logger.log(f'NaNs present ?: {data.isnull().values.any()}')
                nan_count = data.isnull().sum().sum()
                logger.log(f'Total NaNs present: {nan_count}')
                
                logger.log(f'Data size before date-time formating: {len(data) / 10 ** 6:.3f} million rows')
                
                logger.log(f'Data in which the aggregate power is less than the main power for house {house_number}')
                df_anomaly = data[data['aggregate'] < data[appliance_name]]
                df_anomaly.reset_index(inplace=True)
                anomaly_count = len(df_anomaly)
                logger.log(f'Anomalies present for house {house_number}: {anomaly_count} rows out of {len(data) / 10 ** 6}M rows')
                
                del df_anomaly,anomaly_count
                
                data['time'] = pd.to_numeric(data['time'], errors='coerce')
                data['time'] = pd.to_datetime(data['time'], unit='s')
                
                # Check for 'NaT' values in 'time' and drop them
                if data['time'].isnull().any():
                    print("Warning: Some 'NaT' values found in 'time'. Dropping these rows.")
                    data = data.dropna(subset=['time'])
                    logger.log(f'Data size after date-time formating and  Dropping NaT rows : {len(data) / 10 ** 6:.3f} million rows')

                data.set_index('time', inplace=True)
                
                logger.log('resampling Data')
                data = data.resample(str(sample_seconds) + 'S').mean().ffill(limit=1)
                
                
                logger.log(f'Data size after resampling : {len(data) / 10 ** 6:.3f} million rows')
                logger.log("Checking for number of NaNs in the dataframes after resampling")
                logger.log(f'NaNs present ?: {data.isnull().values.any()}')
                nan_count = data.isnull().sum().sum()
                logger.log(f'Total NaNs present: {nan_count}')
                
                logger.log('removing NaNs')
                data = data.dropna()
                logger.log(f'Data size after removing NaNs : {len(data) / 10 ** 6:.3f} million rows')
                if house_number == params_appliance[appliance_name]['test_house']:
                    
                    logger.log(f'Processing file: {filename} for testing')
        
                    
                    # logger.log(data.iloc[:, 0].describe())
                    # data.iloc[:, 0].hist()
                    # plt.show()
                    # logger.log(data.iloc[:, 1].describe())
                    # data.iloc[:, 1].hist()
                    # plt.show()
                    plot_name = f'{appliance_name} - Test Set (House {house_number})'
                    
                    save_plot(data, appliance_name, plot_name, args.save_path,logger)
                    total_length_TT = total_length + len(data)
                    fname = os.path.join(save_path, f'{appliance_name}_test_.csv')
                    #print(data)
                    if saveit:
                        data.to_csv(fname, index=False)
                        logger.log(f'saved test house {house_number} to {fname}')
                
                else:
                    logger.log(f'Processing file: {filename} for trainning and validation')
                    numbers.append(house_number)
                    
                    #total_length_TR = total_length + len(data)
                    #fname = os.path.join(save_path, f'{appliance_name}_training_.csv')
                    train = pd.concat([train, data], ignore_index=True)
                    
                    logger.log(f'added {filename} to trainning/validation set')
                
                del data
                    
            #except: pass
            
            
    logger.log('from trainning/validation set')
    logger.log(f'Total Training/Validation set size: {len(train)/ 10 ** 6:.3f} million rows')
    
    #validation 
    
    logger.log('Processing file for validation')
    
    
    
    
    # Validation crop
    val_len = int((len(train)/100)*validation_percent)
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    
    discib(val,house_number,appliance_name,logger)
    
    #logger.log(val.iloc[:, 0].describe())
    #val.iloc[:, 0].hist()
    #plt.show()
    #logger.log(data.iloc[:, 1].describe())
    #val.iloc[:, 1].hist()
    #plt.show()
    
    plot_name = f'{appliance_name} - Validation Set (House {numbers})'
    total_length_VL = total_length + len(val)
    save_plot(val, appliance_name, plot_name, args.save_path,logger)
    fname = os.path.join(save_path, f'{appliance_name}_validation_.csv')
    
    #print(data)
    if saveit:
        val.to_csv(fname, index=False)
        logger.log(f'saved validation house {numbers} to {fname}')


    #Training  
    train.drop(train.index[-val_len:], inplace=True)    
        
    train.reset_index(drop=True, inplace=True)
    
    logger.log(f'Processing houses: {numbers} ploting for trainning')        
    fname = os.path.join(save_path, f'{appliance_name}_training_.csv') 
    
    logger.log(train['aggregate'].describe())
    #train.iloc[:, 0].hist()
    #plt.show()
    logger.log(train[appliance_name].describe())
    #train.iloc[:, 1].hist()
    #plt.show()
    discib(train,numbers,appliance_name,logger)
    #print(train)
    if saveit: 
        train.to_csv(fname, mode='a', index=False, header=not os.path.isfile(fname))
        logger.log(f'saved training house {numbers} to {fname}')
    
    
    
    plot_name = f'{appliance_name} - Trainng Set (House {numbers})'
    #save_plot(train, appliance_name, plot_name, args.save_path,logger)
    logger.log(f'Total Testing set size: {total_length_TT / 10 ** 6:.3f} million rows')
    logger.log(f'Total Validation set size: {total_length_VL / 10 ** 6:.3f} million rows') 
    logger.log(f'Total training set size: {len(train) / 10 ** 6:.3f} million rows')
    
    logger.log(f'Training, validation, and test sets saved in: {save_path}')
    logger.log(f'Total elapsed time: {(time.time() - start_time) / 60:.2f} minutes')
    
    
    
    del train, numbers, total_length_TT,total_length_VL
    
    logger.log(f'preprocessing ended for {appliance_name} ')
    logger.closefile()


if __name__ == '__main__':
    for app in app_choices:
        
        # Constants
        DATA_DIRECTORY = 'C:/Users/ID/nilmtk_test/mynilm/REFIT/'
        SAVE_DIRECTORY = "C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/refit/raw/"
        
        app_name = app
        
        def get_arguments():
            parser = argparse.ArgumentParser(description='Create new datasets for training')
            parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY, help='Directory containing CLEAN REFIT data')
            parser.add_argument('--appliance_name', type=str, default=app_name, help='Appliance name for training')
            parser.add_argument('--save_path', type=str, default=SAVE_DIRECTORY, help='Directory to store the training data')
            return parser.parse_args()

        main()