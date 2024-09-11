# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:11:55 2024

@author: ID
"""
from ukdale_parameters import *
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
from functions import load_dataframe


DATA_DIRECTORY = 'C:/Users/ID/nilmtk_test/mynilm/UK-DALE/'



def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                          help='The directory containing the UKDALE data')
    parser.add_argument('--appliance_name', type=str, default='fridge',
                          help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')

    return parser.parse_args()


args = get_arguments()
appliance_name = args.appliance_name
print(appliance_name)

sample_seconds = 8


def main():



    #train = pd.DataFrame(columns=['aggregate', appliance_name])

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

        #mains_df['time'] = pd.to_datetime(mains_df['time'], unit='s')
        mains_df.set_index('time', inplace=True)
        mains_df.columns = ['aggregate']
        #resample = mains_df.resample(str(sample_seconds) + 'S').mean()
        mains_df.reset_index(inplace=True)

        #for value in mains_df, app_df:
            

        # the timestamps of mains and appliance are not the same, we need to align them
        # 1. join the aggragte and appliance dataframes;
        # 2. interpolate the missing values;
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)

        df_align = mains_df.join(app_df, how='outer')
        print('converting time to date time')
        
        df_align.index = pd.to_numeric(df_align.index, errors='coerce')
        df_align.index = pd.to_datetime(df_align.index, unit='s')
        
        print('done converting time to date time')
        
        df_align = df_align.resample(str(sample_seconds) + 'S').mean().bfill(limit=1)
        

        df_align.reset_index(inplace=True)
        
        df_align_count = df_align.isnull().sum().sum()
        print(f'NaN values: {df_align_count}')
        
        print('remove NaN values')
        df_align = df_align.dropna()
        
        print("Size of after of Data removing NaNs is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
         
        # print(f'data in which the aggregate power is less than the main power for house {h}')
        
        # df_anomaly = df_align[df_align['aggregate'] < df_align[appliance_name]]
        # df_anomaly.reset_index(inplace=True)
        # print(df_anomaly)
        
        # anomaly_count = len(df_anomaly)
        # print(f'Total anomalies present: {anomaly_count} rows out of {len(df_align) / 10 ** 6}M rows')
        
        # df_align = df_align[~(df_align['aggregate'] < df_align[appliance_name])]
        print("Size of after of Data removing anomalies is {:.4f} M rows.".format(len(df_align)/ 10 ** 6))
        del mains_df, app_df, df_align['time']

        

if __name__ == '__main__':
    main()
