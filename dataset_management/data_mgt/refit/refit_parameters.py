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
