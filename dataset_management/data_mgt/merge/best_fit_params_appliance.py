# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 22:57:28 2024

@author: ID
"""

best_fit_params_appliance = {
    'kettle': {
        'window_length': 599,
        'on_power_threshold': 1500.0,    # Realistic lower threshold for kettle activation.
        'max_on_power': 3100.0,          # Upper bound, balancing between both datasets.
        'min_on_duration': 30.0,         # Realistic duration to avoid very short spikes.
        'min_off_duration': 30.0,        # To ensure distinction between on/off states.
        'alt_app_mean': 700.0,           # Standardized mean.
        'alt_app_std': 1000.0,           # Standardized deviation.
        'c0': 1.0,                       # Loss multiplier (kept consistent).
        's2s_length': 128,               # Sequence length for sequence-to-sequence.
    },
    'microwave': {
        'window_length': 599,
        'on_power_threshold': 300.0,     # Microwaves often start drawing power around this level.
        'max_on_power': 2000.0,          # A mid-range max power to avoid overestimation.
        'min_on_duration': 45.0,         # Average time for heating cycles.
        'min_off_duration': 30.0,        # Ignore brief off periods between cycles.
        'alt_app_mean': 475.0,           # Balanced mean for both datasets.
        'alt_app_std': 775.0,            # Adjusted standard deviation.
        'c0': 1.0,                       # Loss multiplier.
        's2s_length': 128,               # Sequence length for sequence-to-sequence.
    },
    'fridge': {
        'window_length': 599,
        'on_power_threshold': 40.0,      # Fridges have low activation thresholds.
        'max_on_power': 250.0,           # Maximum power that avoids outliers.
        'min_on_duration': 300.0,        # Fridges often stay on for long periods.
        'min_off_duration': 50.0,        # Short off periods to account for compressor cycles.
        'alt_app_mean': 200.0,           # Standardized mean.
        'alt_app_std': 400.0,            # Standardized deviation.
        'c0': 1e-06,                     # Loss multiplier.
    },
    'dishwasher': {
        'window_length': 599,
        'on_power_threshold': 20.0,      # Dishwashers have low power draw during certain phases.
        'max_on_power': 2550.0,          # Maximum power based on both datasets.
        'min_on_duration': 1500.0,       # Dishwashers run for long periods.
        'min_off_duration': 1600.0,      # Ensure distinct on/off periods.
        'alt_app_mean': 700.0,           # Standardized mean.
        'alt_app_std': 1000.0,           # Standardized deviation.
        'c0': 1.0,                       # Loss multiplier.
    },
    'washingmachine': {
        'window_length': 599,
        'on_power_threshold': 50.0,      # Increased threshold for capturing active washing cycles.
        'max_on_power': 2600.0,          # Realistic max power based on common usage.
        'min_on_duration': 1200.0,       # Average washing machine cycle duration.
        'min_off_duration': 600.0,       # Ignore brief pauses in wash cycles.
        'alt_app_mean': 400.0,           # Standardized mean.
        'alt_app_std': 700.0,            # Standardized deviation.
        'c0': 1e-02,                     # Loss multiplier.
    }
}


# Justification for Adjustments:

# Kettle:

# A lower threshold of 1500W is realistic for activation since kettles generally draw substantial power but not too high to miss certain models.
# The max_on_power remains 3000W to capture high-end kettles.

# Microwave:

# A threshold of 300W is chosen to account for microwaves operating at lower power modes (defrosting, etc.).
# A max_on_power of 2000W avoids overestimation, which aligns with most household microwaves.

# Fridge:

# A low threshold of 40W is typical for fridge compressors starting up.
# The max_on_power of 250W ensures the model doesn't overestimate fridge power consumption.

# Dishwasher:

# The on_power_threshold remains low (20W), capturing various phases of dishwasher operation.
# Max_on_power is set to 2550W, balancing the range of dishwashers from both datasets.
# Washing Machine:

# A higher threshold of 50W captures the washing cycle without false negatives.
# Max_on_power is capped at 2600W to accommodate most household washing machines.
# These values are more balanced and reflect typical power usage patterns, ensuring that the CNN model can generalize well across both datasets while capturing essential appliance behavior.







