import pandas as pd


def load(path, building, appliance, channel):

    # load csv
    file_name = path + 'CLEAN_House' + str(building) + '.csv'
    single_csv = pd.read_csv(file_name,
                             header=0,
                             names=['aggregate', appliance],
                             usecols=[2, channel+2],
                             na_filter=False,
                             parse_dates=True,
                             infer_datetime_format=True,
                             memory_map=True
                             )

    return single_csv