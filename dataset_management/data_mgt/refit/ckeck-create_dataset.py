# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:18:12 2024

@author: ID
"""



import time
import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import common
import socket

# Constants
app_choices=['kettle']#, 'microwave', 'fridge', 'dishwasher', 'washingmachine']
#app_name = 'washingmachine'
saveit = False
plot =False


# Appliance-specific parameters
params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 19, 20],
        'channels': [8, 9, 9, 8, 7, 9, 9, 7, 6, 9, 5, 9],
        'test_house': 2,
        'validation_house': 5,
        'test_on_train_house': 5,
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [4, 10, 12, 17, 19],
        'channels': [8, 8, 3, 7, 4],
        'test_house': 4,
        'validation_house': 17,
        'test_on_train_house': 10,
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [2, 5, 9, 12, 15],
        'channels': [1, 1, 1, 1, 1],
        'test_house': 15,
        'validation_house': 12,
        'test_on_train_house': 5,
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [5, 7, 9, 13, 16, 18, 20],
        'channels': [4, 6, 4, 4, 6, 6, 5],
        'test_house': 20,
        'validation_house': 18,
        'test_on_train_house': 13,
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [2, 5, 7, 8, 9, 15, 16, 17, 18],
        'channels': [2, 3, 5, 4, 3, 3, 5, 4, 5],
        'test_house': 8,
        'validation_house': 18,
        'test_on_train_house': 5,
    }
}


def load_data(path, building, appliance, channel):
    file_name = os.path.join(path, f'CLEAN_House{building}.csv')
    return pd.read_csv(file_name, header=0, names=['aggregate', appliance], usecols=[2, channel+2], parse_dates=True, memory_map=True)

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
        plot_filepath = os.path.join(save_dir,'outputs_log', appliance_name, 'images', f'{plot_name}.png')
        os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
        if saveit: plt.savefig(plot_filepath)
        plt.show()
        plt.close()

def discib(data,house_number,appliance_name):
    
    print("check for number of NaNs in the dataframes")
    print(f'NaNs present ?: {data.isnull().values.any()}')
    nan_count = data.isnull().sum().sum()
    print(f'Total NaNs present: {nan_count}')
    
    print(f'data in which the aggregate power is less than the main power for house {house_number}')
    
    df_anomaly = data[data['aggregate'] < data[appliance_name]]
    df_anomaly.reset_index(inplace=True)
    print(df_anomaly)
    
    anomaly_count = len(df_anomaly)
    max_on_power = params_appliance[appliance_name]['max_on_power']
    
    print(f'anomalies present for house {house_number}: {anomaly_count} rows out of {len(data) / 10 ** 6}M rows')
    # Limit appliance power to [0, max_on_power].
    print(f'Limiting appliance power to [0, {max_on_power}]')
    
    data.loc[:, appliance_name] = data.loc[:, appliance_name].clip(0, max_on_power)
    # Get appliance status and add to end of dataframe.
    print('Computing on-off status.')
    status = common.compute_status(data.loc[:, appliance_name].to_numpy(), appliance_name)
    data.insert(2, 'status', status)
    
    df_on = data[data['status']==1]
    num_on = len(df_on)
    df_off = data[data['status']==0]
    num_off = len(df_off)
    num_total = data.iloc[:, 2].size
    #print(df_on)
    print(f'Number of samples with on status: {num_on}.')
    #print(df_on.loc[:, appliance].describe())
    #print(df_off)
    print(f'Number of samples with off status: {num_off}.')
    print(f'Number of total samples: {num_total}.')
      
def main():
    start_time = time.time()
    args = get_arguments()
    

    appliance_name = args.appliance_name
    appliance  = args.appliance_name
    
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
                    print(f'Processing file: {filename}')
                    data = load_data(path, house_number, appliance_name, params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(house_number)])
                    
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
                        if saveit: data.to_csv(fname, index=False)
                    elif house_number == params_appliance[appliance_name]['validation_house']:
                        print(f'Processing file: {filename} for validation')
                        
                        discib(data,house_number,appliance_name)
                        
                        #print(data.iloc[:, 0].describe())
                        #data.iloc[:, 0].hist()
                        #plt.show()
                        #print(data.iloc[:, 1].describe())
                        #data.iloc[:, 1].hist()
                        #plt.show()
                        
                        plot_name = f'{appliance_name} - Validation Set (House {house_number})'
                        total_length_VL = total_length + len(data)
                        save_plot(data, appliance_name, plot_name, args.save_path)
                        fname = os.path.join(save_path, f'{appliance_name}_validation_.csv')
                        if saveit: data.to_csv(fname, index=False)
                    else:
                        print(f'Processing file: {filename} for trainning')
                        numbers.append(house_number)
                        print(f'added {filename} to trainning set')
                        total_length_TR = total_length + len(data)
                        fname = os.path.join(save_path, f'{appliance_name}_training_.csv')
                        if saveit: data.to_csv(fname, mode='a', index=False, header=not os.path.isfile(fname))
                        
                        train = pd.concat([train, data], index= False)
            
            except:
                pass
            
        
           
            
    print(f'Processing houses: {numbers} ploting for trainning')        
    # fname = os.path.join(save_path, f'{appliance_name}_training_.csv') 
    # data = pd.read_csv(fname)
    print(train['aggregate'].describe())
    #data.iloc[:, 0].hist()
    #plt.show()
    print(train[appliance_name].describe())
    #data.iloc[:, 1].hist()
    #plt.show()
    
    discib(train,numbers,appliance_name)
    
    plot_name = f'{appliance_name} - Trainng Set (House {numbers})'
    save_plot(data, appliance_name, plot_name, args.save_path)
    print(f'Total Testing set size: {total_length_TT / 10 ** 6:.3f} million rows')
    print(f'Total Validation set size: {total_length_VL / 10 ** 6:.3f} million rows') 
    print(f'Total training set size: {len(train) / 10 ** 6:.3f} million rows')
    print(f'Training, validation, and test sets saved in: {save_path}')
    print(f'Total elapsed time: {(time.time() - start_time) / 60:.2f} minutes')
    
    
    
    del data,train, numbers, total_length_TT,total_length_VL, total_length_TR

if __name__ == '__main__':
    for app in app_choices:
        
        # Constants
        DATA_DIRECTORY = 'C:/Users/ID/nilmtk_test/mynilm/REFIT/'
        SAVE_DIRECTORY = "C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/refit/"
        
        app_name = app
        
        def get_arguments():
            parser = argparse.ArgumentParser(description='Create new datasets for training')
            parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY, help='Directory containing CLEAN REFIT data')
            parser.add_argument('--appliance_name', type=str, default=app_name, help='Appliance name for training')
            parser.add_argument('--save_path', type=str, default=SAVE_DIRECTORY, help='Directory to store the training data')
            return parser.parse_args()

        main()
