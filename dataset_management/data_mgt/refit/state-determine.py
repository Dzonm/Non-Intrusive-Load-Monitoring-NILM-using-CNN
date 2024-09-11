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

import socket

import common

# Constants
app_choices=['kettle']#, 'microwave', 'fridge', 'dishwasher', 'washingmachine']
#app_name = 'washingmachine'

saveit = False
plot =False
training_building_percent = 80
validation_percent = 20



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

def load_data(path, building, appliance, channel):
    file_name = os.path.join(path, f'CLEAN_House{building}.csv')
    
    # Assuming the first column is 'timestamp' and you want to use it as the index
    return pd.read_csv(file_name, header=0, names=['timestamp', 'aggregate', appliance],
                       usecols=[0, 2, channel+2], parse_dates=['timestamp'], index_col='timestamp', memory_map=True)

def compute_stats(df):
    return {
        'mean': df.mean(),
        'std': df.std(),
        'median': df.median(),
        'quartile1': df.quantile(q=0.25),
        'quartile3': df.quantile(q=0.75)
    }

def save_plot(data, appliance_name, plot_name, save_dir):
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
        print(f'Processing plot for : {appliance_name}')
        plot_filepath = os.path.join(save_dir,'outputs_log', appliance_name, f'{plot_name}.png')
        os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
        if saveit: 
            plt.savefig(plot_filepath)
            print(f'saved plot for : {appliance_name} to {plot_filepath}')
        plt.show()
        plt.close()
    
    
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
        
        # Ensure the index is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.train_data.index):
            self.train_data.index = pd.to_datetime(self.train_data.index)

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

# Inside the main() function after data loading and processing:
def discib(data, house_number, appliance_name):
    print("Checking for number of NaNs in the dataframes")
    print(f'NaNs present ?: {data.isnull().values.any()}')
    nan_count = data.isnull().sum().sum()
    print(f'Total NaNs present: {nan_count}')
    
    print(f'Data in which the aggregate power is less than the main power for house {house_number}')
    
    df_anomaly = data[data['aggregate'] < data[appliance_name]]
    df_anomaly.reset_index(inplace=True)
    print(df_anomaly)
    
    anomaly_count = len(df_anomaly)
    max_on_power = params_appliance[appliance_name]['max_on_power']
    
    print(f'Anomalies present for house {house_number}: {anomaly_count} rows out of {len(data) / 10 ** 6}M rows')
    # Limit appliance power to [0, max_on_power].
    
    df = pd.DataFrame(columns=['aggregate', appliance_name])
    df = pd.concat([df, data], ignore_index=True)
    
    df.loc[:, appliance_name] = df.loc[:, appliance_name].clip(0, max_on_power)
    
    # Get appliance status and add to end of dataframe.
    # print('Computing on-off status.')
    # status = common.compute_status(df.loc[:, appliance_name].to_numpy(), appliance_name)
    # df.insert(2, 'status', status)

    print('Compute appliance durations')
    duration_calculator = ApplianceDuration(df, appliance_name, params_appliance[appliance_name]['on_power_threshold'])
    min_on_duration = duration_calculator.min_on_duration()
    min_off_duration = duration_calculator.min_off_duration()
    
    print(f'Minimum ON duration for {appliance_name}: {min_on_duration} seconds')
    print(f'Minimum OFF duration for {appliance_name}: {min_off_duration} seconds')
    del df_anomaly,nan_count


def main():
    start_time = time.time()
    args = get_arguments()
    
    appliance = args.appliance_name
    appliance_name = args.appliance_name
    
    
    print('start pre-processing')
    print(f'Machine name: {socket.gethostname()}')
    print(args)

    print(f'Target appliance: {appliance}')
    
    appliance_name = args.appliance_name
    path = args.data_dir
    save_path = os.path.join(args.save_path, appliance_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f'Processing appliance: {appliance_name}')
    print(f'Data directory: {path}')
    print(f'Save directory: {save_path}')
    
    total_length = 0
    numbers = []
    
    train = pd.DataFrame(columns=['aggregate', appliance_name])
    
    for filename in os.listdir(path):
        if "CLEAN_House" in filename:
            try:
                house_number = int(re.search(r'\d+', filename).group())
                
                if house_number in params_appliance[appliance_name]['houses']:
                    
                    chnl = params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(house_number)]
                    data = load_data(path, house_number, appliance_name, chnl)
                    
                    print(f'Processing file: {filename}, {appliance_name} - channel {chnl}')
                    #print(data)
                    if house_number == params_appliance[appliance_name]['test_house']:
                        
                        print(f'Processing file: {filename} for testing')
                        
                        discib(data,house_number,appliance_name)
                        
                        # print(data.iloc[:, 0].describe())
                        # data.iloc[:, 0].hist()
                        # plt.show()
                        # print(data.iloc[:, 1].describe())
                        # data.iloc[:, 1].hist()
                        # plt.show()
                        plot_name = f'{appliance_name} - Test Set (House {house_number})'
                        
                        save_plot(data, appliance_name, plot_name, args.save_path)
                        total_length_TT = total_length + len(data)
                        fname = os.path.join(save_path, f'{appliance_name}_test_.csv')
                        #print(data)
                        if saveit:
                            data.to_csv(fname, index=False)
                            print(f'saved test house {house_number} to {fname}')
                    
                    else:
                        print(f'Processing file: {filename} for trainning and validation')
                        numbers.append(house_number)
                        print(f'added {filename} to trainning/validation set')
                        #total_length_TR = total_length + len(data)
                        #fname = os.path.join(save_path, f'{appliance_name}_training_.csv')
                        train = pd.concat([train, data], ignore_index=True)
                    
                    del data
                        
            except:
                pass
            
            
    print('from trainning/validation set')
    print(f'Total Training/Validation set size: {len(train)/ 10 ** 6:.3f} million rows')
    
    #validation 
    
    print('Processing file for validation')
    
    
    # Validation crop
    val_len = int((len(train)/100)*validation_percent)
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    
    discib(val,house_number,appliance_name)
    
    #print(val.iloc[:, 0].describe())
    #val.iloc[:, 0].hist()
    #plt.show()
    #print(data.iloc[:, 1].describe())
    #val.iloc[:, 1].hist()
    #plt.show()
    
    plot_name = f'{appliance_name} - Validation Set (House {numbers})'
    total_length_VL = total_length + len(val)
    save_plot(val, appliance_name, plot_name, args.save_path)
    fname = os.path.join(save_path, f'{appliance_name}_validation_.csv')
    
    #print(data)
    if saveit:
        val.to_csv(fname, index=False)
        print(f'saved validation house {numbers} to {fname}')


    #Training  
    train.drop(train.index[-val_len:], inplace=True)    
        
    train.reset_index(drop=True, inplace=True)
    
    print(f'Processing houses: {numbers} ploting for trainning')        
    fname = os.path.join(save_path, f'{appliance_name}_training_.csv') 
    
    print(train['aggregate'].describe())
    #train.iloc[:, 0].hist()
    #plt.show()
    print(train[appliance_name].describe())
    #train.iloc[:, 1].hist()
    #plt.show()
    discib(train,numbers,appliance_name)
    #print(train)
    if saveit: 
        train.to_csv(fname, mode='a', index=False, header=not os.path.isfile(fname))
        print(f'saved training house {numbers} to {fname}')
    
    
    
    plot_name = f'{appliance_name} - Trainng Set (House {numbers})'
    #save_plot(train, appliance_name, plot_name, args.save_path,logger)
    print(f'Total Testing set size: {total_length_TT / 10 ** 6:.3f} million rows')
    print(f'Total Validation set size: {total_length_VL / 10 ** 6:.3f} million rows') 
    print(f'Total training set size: {len(train) / 10 ** 6:.3f} million rows')
    
    print(f'Training, validation, and test sets saved in: {save_path}')
    print(f'Total elapsed time: {(time.time() - start_time) / 60:.2f} minutes')
    
    
    
    del train, numbers, total_length_TT,total_length_VL
    
    print(f'preprocessing ended for {appliance_name} ')



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