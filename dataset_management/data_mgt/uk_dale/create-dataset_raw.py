from ukdale_parameters import params_appliance
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
from functions import load_dataframe
import os
from logger import Logger
import socket


# Constants
app_choices=['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
#app_name = 'kettle'



def main():

    start_time = time.time()
    training_building_percent = 80
    validation_percent = 20
    sample_seconds = 8
    plot = True
    saveit = True

    train = pd.DataFrame(columns=['aggregate', appliance_name])
    
    

    for h in params_appliance[appliance_name]['houses']:
        logger.log('    ' + args.data_dir + 'house_' + str(h) + '/'
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
        
        logger.log(f'==========checking for NaN values for house {h}==============')
        main_nan_count = mains_df.isnull().sum().sum()
        logger.log(f'mains NaN values: {main_nan_count}')
        app_nan_count = app_df.isnull().sum().sum()
        logger.log(f'appliances NaN values: {app_nan_count}')
        logger.log(" Size of total mains set is {:.4f} M rows.".format(len(mains_df) / 10 ** 6))
        logger.log(" Size of total appliance set is {:.4f} M rows.".format(len(app_df) / 10 ** 6))
        
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
                    
            plot_filepath = os.path.join(Save_directory, 'outputs_log',f'{appliance_name}', f'{plot_name}')
            logger.log(f'Plot directory of {plot_name}: {plot_filepath}')
            if saveit: plt.savefig(fname=plot_filepath)
            plt.show()
            plt.close()

        # Appliance
        app_df['time'] = pd.to_numeric(app_df['time'], errors='coerce')
        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')

        if plot:
            logger.log(f'plot of {appliance_name} for house {h}')
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
            
            plot_filepath = os.path.join(Save_directory,'outputs_log',f'{appliance_name}', f'{plot_name}')
            logger.log(f'Plot directory of {plot_name}: {plot_filepath}')
            if saveit: plt.savefig(fname=plot_filepath)
            plt.show()
            plt.close()

        # the timestamps of mains and appliance are not the same, we need to align them
        # 1. join the aggragte and appliance dataframes;
        # 2. interpolate the missing values by forward fill;
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)
        
        df_align = mains_df.join(app_df, how='outer').resample(str(sample_seconds) + 'S').mean().ffill(limit=1)
        
        
        
        logger.log(f'Data in which the aggregate power is less than the main power for house {h}')
        df_anomaly = df_align[df_align['aggregate'] < df_align[appliance_name]]
        logger.log(df_anomaly)
        anomaly_count = len(df_anomaly)
        logger.log(f'Total anomalies present: {anomaly_count} rows out of {len(df_align) / 10 ** 6}M rows')
        
        
        
        # Check for NaNs.     
        logger.log("check for number of NaNs after combining the aggregate and appliance dataframes")
        logger.log(f'NaNs present ?: {df_align.isnull().values.any()}')
        nan_count = df_align.isnull().sum().sum()
        logger.log(f'Total NaNs present: {nan_count}')
             
        logger.log('removing NaNs')
        df_align = df_align.dropna()
        
        logger.log("Check for NaNs affer removing NaNs")     
        logger.log(f"check if there are NaNs?:{df_align.isnull().values.any()}")
        nan_count = df_align.isnull().sum().sum()
        logger.log(f'Total NaNs present: {nan_count}')
        logger.log(" Size of Data after removing NaNs is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
        

        df_align.reset_index(inplace=True)


        del mains_df, app_df, df_align['time'],df_anomaly
        
            

        if h == params_appliance[appliance_name]['test_build']:
            # Test CSV
            if plot:
                logger.log(f'Aggregate, {appliance_name} and status PLOT FOR HOUSE {h} used for TESTING')
                
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
                
                plot_filepath = os.path.join(Save_directory,'outputs_log',f'{appliance_name}', f'{plot_name}')
                logger.log(f'Plot directory of {plot_name}: {plot_filepath}')
                if saveit: plt.savefig(fname=plot_filepath)
                plt.show()
                plt.close()
            
            df_align.to_csv(args.save_path + appliance_name +'_test_.csv',index=False)
            logger.log(" Size of test set is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
            logger.log(f'house{h} dataset: use for testing')
            logger.log(df_align['aggregate'].describe())
            logger.log(df_align[appliance_name].describe())
            test= df_align
            
            continue
        
        train = pd.concat([train, df_align], ignore_index=True)
        
        
        del df_align

       
    #house for train & validation
    h = params_appliance[appliance_name]['train_build']
    
    
    logger.log(f'house{h} dataset: use for tranning and validation')
    logger.log(" Size of tranning and validation set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    logger.log(train['aggregate'].describe())
    logger.log(train[appliance_name].describe())
    
   
    
    # Validation crop
    val_len = int((len(train)/100)*validation_percent)
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    # Validation CSV
    val.to_csv(args.save_path + appliance_name + '_validation_' + '.csv',index=False)
    
    
    
    #validation plot
    if plot:
        logger.log(f'Aggregate, {appliance_name} and status house {h} which is used for VALIDATION')
        
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
    
        
        
        plot_filepath = os.path.join(Save_directory,'outputs_log',f'{appliance_name}', f'{plot_name}')
        logger.log(f'Plot directory of {plot_name}: {plot_filepath}')
        if saveit: plt.savefig(fname=plot_filepath)
        plt.show()
        plt.close()
    # Training crop
    train.drop(train.index[-val_len:], inplace=True)    
        
    train.reset_index(drop=True, inplace=True)
        
    # Training CSV
    train.to_csv(args.save_path + appliance_name + '_training_.csv',index=False)
    if plot:
        logger.log(f'Aggregate, {appliance_name} and status of house {h} - TRAINING')
        
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
        
        plot_filepath = os.path.join(Save_directory,'outputs_log',f'{appliance_name}', f'{plot_name}')
        logger.log(f'Plot directory of {plot_name}: {plot_filepath}')
        if saveit: plt.savefig(fname=plot_filepath)
        plt.show()
        plt.close()
    
    logger.log("    Size of total training set is {:.4f} M rows.".format(len(test) / 10 ** 6))
    logger.log("    Size of total training set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    logger.log("    Size of total validation set is {:.4f} M rows.".format(len(val) / 10 ** 6))
    del train, val, test


    logger.log("\nPlease find files in: " + args.save_path)
    logger.log("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    logger.log('Processing completed.')
    logger.closefile()


if __name__ == '__main__':
    for app in app_choices:
        app_name = app
        DATA_DIRECTORY = 'C:/Users/ID/nilmtk_test/mynilm/UK-DALE/'
        Save_directory = "C:/Users/ID/nilmtk_test/mynilm/COMBINE_UKDALE+REFIT/uk_dale/raw/"
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
        
        logger = Logger(
            log_file_name = os.path.join(
                Save_directory,'outputs_log', appliance_name, f'{appliance_name}_raw_create_log.log'
            )
        )
        
        logger.log('Processing started.')
        logger.log(f'Machine name: {socket.gethostname()}')
        logger.log(args)
        main()
        
        

        