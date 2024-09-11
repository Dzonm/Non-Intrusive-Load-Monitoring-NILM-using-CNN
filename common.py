

import os
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

# Alternative aggregate standardization parameters used for all appliances.
# From Michele D’Incecco, et. al., "Transfer Learning for Non-Intrusive Load Monitoring"
ALT_AGGREGATE_MEAN = 548.8655574846409  # in Watts
ALT_AGGREGATE_STD = 831.5511096207229   # in Watts

# If True the alternative standardization parameters will be used
# for scaling the datasets.
USE_ALT_STANDARDIZATION = False

# If True the appliance dataset will be normalized to [0, max_on_power]
# else the appliance dataset will be z-score standardized.
USE_APPLIANCE_NORMALIZATION = False

# Power consumption sample update period in seconds.
SAMPLE_PERIOD = 8

# Various parameters used for training, validation and testing.
# Except where noted, values are calculated from statistical analysis
# of the respective dataset.
params_appliance = {
    'kettle': {
        'window_length': 599,  # General setting for all appliances

        # Lower threshold to capture low-power modes or initial heating.
        'on_power_threshold': 1500.0,    # Realistic lower threshold for kettle activation.
         
        # Adjusted max power to prevent overestimation.
        'max_on_power': 3100.0,          # Upper bound, balancing between both datasets.
         
        # Increased duration to avoid false positives from brief heating cycles.
        'min_on_duration': 30.0,         # Realistic duration to avoid very short spikes.
         
        # Increase to ignore brief power drops.
        'min_off_duration': 30.0,        # To ensure distinction between on/off states.
         
        # Ensure correct normalization based on training data.
        'train_agg_mean': 548.8655574846409,
        'train_agg_std': 831.5511096207229,
        'train_app_mean': 19.87566415366713,
        'train_app_std': 214.25969704679514,
         
        # Test dataset characteristics.
        'test_app_mean': 17.2624718893315,
        'test_agg_mean': 360.9094939576277,
         
        # Alternative standardization parameters.
        'alt_app_mean': 700.0,
        'alt_app_std': 1000.0,
         
        # L1 Loss multiplier fine-tuning.
        'c0': 1.0,
         
        's2s_length': 128,  # Sequence length for sequence-to-sequence modeling
    },
    'microwave': {
        'window_length': 599,  # General setting for all appliances
        
        # Adjusted to capture lower power modes and reduce false negatives.
        'on_power_threshold': 300.0,     # Microwaves often start drawing power around this level.
        
        # Adjusted max power to avoid overestimation.
        'max_on_power': 2000.0,          # A mid-range max power to avoid overestimation.
        
        # Increased to filter out short power spikes.
        'min_on_duration': 45.0,         # Average time for heating cycles.
        
        # Increased to distinguish actual "off" periods.
        'min_off_duration': 30.0,        # Ignore brief off periods between cycles.
        
        # Normalization based on training data.
        'train_agg_mean': 548.6746553054109,
        'train_agg_std': 831.3604408749264,
        'train_app_mean': 8.825710060041486,
        'train_app_std': 110.49172419185511,
        
        # Test dataset characteristics.
        'test_app_mean': 7.1218760786987145,
        'test_agg_mean': 361.2540948167443,
        
        # Alternative standardization parameters.
        'alt_app_mean': 475.0,
        'alt_app_std': 775.0,
        
        # L1 Loss multiplier fine-tuning.
        'c0': 1.0,
        
        's2s_length': 128,  # Sequence length for sequence-to-sequence modeling
    },
    'fridge': {
        'window_length': 599,
        'on_power_threshold': 40.0,      # Fridges have low activation thresholds.
        'max_on_power': 250.0,           # Maximum power that avoids outliers.
        'min_on_duration': 300.0,        # Fridges often stay on for long periods.
        'min_off_duration': 50.0,        # Short off periods to account for compressor cycles.
        'train_agg_mean': 548.788877622983,
        'train_agg_std': 831.5468664228363,
        'train_app_mean': 37.670549841926395,
        'train_app_std': 48.675252777091224,
        'test_app_mean': 37.84415530366245,
        'test_agg_mean': 361.2457809063843,
        'alt_app_mean': 200.0,
        'alt_app_std': 400.0,
        'c0': 1e-06
    },
    'dishwasher': {
        'window_length': 599,
        'on_power_threshold': 20.0,      # Dishwashers have low power draw during certain phases.
        'max_on_power': 2550.0,          # Maximum power based on both datasets.
        'min_on_duration': 1500.0,       # Dishwashers run for long periods.
        'min_off_duration': 1600.0,      # Ensure distinct on/off periods.
        'train_agg_mean': 547.4509676572064,
        'train_agg_std': 829.4337827830434,
        'train_app_mean': 39.6154427987629,
        'train_app_std': 283.75762008069836,
        'test_app_mean': 18.00129343246976,
        'test_agg_mean': 361.2469978964126,
        'alt_app_mean': 700.0,
        'alt_app_std': 1000.0,
        'c0': 1.0
    },
    'washingmachine': {
        'window_length': 599,
        'on_power_threshold': 50.0,      # Increased threshold for capturing active washing cycles.
        'max_on_power': 2600.0,          # Realistic max power based on common usage.
        'min_on_duration': 1200.0,       # Average washing machine cycle duration.
        'min_off_duration': 600.0,       # Ignore brief pauses in wash cycles.
        'train_agg_mean': 548.582282742119,
        'train_agg_std': 830.3767090984546,
        'train_app_mean': 23.763518597743328,
        'train_app_std': 189.25700089511307,
        'test_app_mean': 8.797156254803742,
        'test_agg_mean': 361.26311668270864,
        'alt_app_mean': 400.0,
        'alt_app_std': 700.0,
        'c0': 1e-02
    }
}
def find_test_filename(test_dir, appliance, test_type) -> str:
    """Find test file name given a datset name."""
    for filename in os.listdir(os.path.join(test_dir, appliance)):
        if test_type == 'train' and 'TRAIN' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'uk' and 'UK' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'redd' and 'REDD' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'test' and 'TEST' in\
                filename.upper() and 'TRAIN' not in filename.upper() and 'UK' not in filename.upper():
            test_filename = filename
            break
        elif test_type == 'val' and 'VALIDATION' in filename.upper():
            test_filename = filename
            break
    return test_filename

def load_dataset(file_name, crop=None):
    """Load CSV file and return mains power, appliance power and status."""
    df = pd.read_csv(file_name, nrows=crop)

    mains_power = np.array(df.iloc[:, 0], dtype=np.float32)
    appliance_power = np.array(df.iloc[:, 1], dtype=np.float32)
    activations = np.array(df.iloc[:, 2], dtype=np.float32)

    return mains_power, appliance_power, activations

def tflite_infer(interpreter, provider, num_eval, eval_offset=0, log=print) -> list:
    """Perform inference using a tflite model"""
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    log(f'interpreter input details: {input_details}')
    output_details = interpreter.get_output_details()
    log(f'interpreter output details: {output_details}')
    # Check I/O tensor type.
    input_dtype = input_details[0]['dtype']
    floating_input = input_dtype == np.float32
    log(f'tflite model floating input: {floating_input}')
    output_dtype = output_details[0]['dtype']
    floating_output = output_dtype == np.float32
    log(f'tflite model floating output: {floating_output}')
    # Get I/O indices.
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    # If model has int I/O get quantization information.
    if not floating_input:
        input_quant_params = input_details[0]['quantization_parameters']
        input_scale = input_quant_params['scales'][0]
        input_zero_point = input_quant_params['zero_points'][0]
    if not floating_output:
        output_quant_params = output_details[0]['quantization_parameters']
        output_scale = output_quant_params['scales'][0]
        output_zero_point = output_quant_params['zero_points'][0]

    # Calculate num_eval sized indices of contiguous locations in provider.
    # Get number of samples per batch in provider. Since batch should always be
    # set to 1 for inference, this will simply return the total number of samples.
    samples_per_batch = len(provider)
    if num_eval - eval_offset > samples_per_batch:
        raise ValueError('Not enough test samples to run evaluation.')
    eval_indices = list(range(samples_per_batch))[eval_offset:num_eval+eval_offset]

    log(f'Running inference on {num_eval} samples...')
    start = time.time()
    def infer(i):
        sample, target, _= provider[i]
        if not sample.any():
            return 0.0, 0.0 # ignore missing data
        ground_truth = np.squeeze(target)
        if not floating_input: # convert float to int
            sample = sample / input_scale + input_zero_point
            sample = sample.astype(input_dtype)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke() # run inference
        result = interpreter.get_tensor(output_index)
        prediction = np.squeeze(result)
        if not floating_output: # convert int to float
            prediction = (prediction - output_zero_point) * output_scale
        #print(f'sample index: {i} ground_truth: {ground_truth:.3f} prediction: {prediction:.3f}')
        return ground_truth, prediction
    results = [infer(i) for i in tqdm(eval_indices)]
    end = time.time()
    log('Inference run complete.')
    log(f'Inference rate: {num_eval / (end - start):.3f} Hz')

    return results

def normalize(dataset):
    """Normalize or standardize a dataset."""
    # Compute aggregate statistics.
    agg_mean = np.mean(dataset[0])
    agg_std = np.std(dataset[0])
    print(f'agg mean: {agg_mean}, agg std: {agg_std}')
    agg_median = np.percentile(dataset[0], 50)
    agg_quartile1 = np.percentile(dataset[0], 25)
    agg_quartile3 = np.percentile(dataset[0], 75)
    print(f'agg median: {agg_median}, agg q1: {agg_quartile1}, agg q3: {agg_quartile3}')
    # Compute appliance statistics.
    app_mean = np.mean(dataset[1])
    app_std = np.std(dataset[1])
    print(f'app mean: {app_mean}, app std: {app_std}')
    app_median = np.percentile(dataset[1], 50)
    app_quartile1 = np.percentile(dataset[1], 25)
    app_quartile3 = np.percentile(dataset[1], 75)
    print(f'app median: {app_median}, app q1: {app_quartile1}, app q3: {app_quartile3}')
    def z_norm(dataset, mean, std):
        return (dataset - mean) / std
    def robust_scaler(dataset, median, quartile1, quartile3): #pylint: disable=unused-variable
        return (dataset - median) / (quartile3 - quartile1)
    return (
        z_norm(
            dataset[0], agg_mean, agg_std),
        z_norm(
            dataset[1], app_mean, app_std))

def compute_status(appliance_power:np.ndarray, appliance:str) -> list:
    """Compute appliance on-off status."""
    threshold = params_appliance[appliance]['on_power_threshold']

    def ceildiv(a:int, b:int) -> int:
        """Upside-down floor division."""
        return -(a // -b)

    # Convert durations from seconds to samples.
    min_on_duration = ceildiv(params_appliance[appliance]['min_on_duration'],
                              SAMPLE_PERIOD)
    min_off_duration = ceildiv(params_appliance[appliance]['min_off_duration'],
                               SAMPLE_PERIOD)

    # Apply threshold to appliance powers.
    initial_status = appliance_power.copy() >= threshold

    # Find transistion indices.
    status_diff = np.diff(initial_status)
    events_idx = status_diff.nonzero()
    events_idx = np.array(events_idx).squeeze()
    events_idx += 1

    # Adjustment for first and last transition.
    if initial_status[0]:
        events_idx = np.insert(events_idx, 0, 0)
    if initial_status[-1]:
        events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

    # Separate out on and off events.
    events_idx = events_idx.reshape((-1, 2))
    on_events = events_idx[:, 0].copy()
    off_events = events_idx[:, 1].copy()
    assert len(on_events) == len(off_events)

    # Filter out on and off transitions faster than minimum values.
    if len(on_events) > 0:
        off_duration = on_events[1:] - off_events[:-1]
        off_duration = np.insert(off_duration, 0, 1000)
        on_events = on_events[off_duration > min_off_duration]
        off_events = off_events[np.roll(off_duration, -1) > min_off_duration]

        on_duration = off_events - on_events
        on_events = on_events[on_duration >= min_on_duration]
        off_events = off_events[on_duration >= min_on_duration]
        assert len(on_events) == len(off_events)

    # Generate final status.
    status = [0] * appliance_power.size
    for on, off in zip(on_events, off_events):
        status[on: off] = [1] * (off - on)

    return status



def post_process_activations(ground_truth_activations, predicted_activations, appliance):
    """
    Post-processing algorithm to eliminate irrelevant activations with predicted status of zero.
    
    Args:
    ground_truth_activations (list): List of ground truth activation arrays
    predicted_activations (list): List of predicted activation arrays
    appliance (str): The appliance type to consider for thresholding
    
    Returns:
    numpy.ndarray: Updated predicted energy profile
    """
    threshold = params_appliance[appliance]['on_power_threshold']

    # Calculate lengths of ground truth activations
    ground_truth_lengths = [len(act) for act in ground_truth_activations]
    
    # Determine minimum length from ground truth activations
    min_length = min(ground_truth_lengths)
    
    # Initialize updated predicted activation profile and energy profile
    updated_predicted_activations = []
    updated_predicted_energy = np.zeros_like(np.concatenate(predicted_activations))
    
    # Compare and eliminate irrelevant activations
    current_index = 0
    for activation in predicted_activations:
        if len(activation) >= min_length:
            # Compute the predicted status based on the threshold
            predicted_status = activation >= threshold
            # Filter out predicted energy with predicted status of zero
            activation_filtered = np.where(predicted_status, activation, 0)
            updated_predicted_activations.append(activation_filtered)
            updated_predicted_energy[current_index:current_index + len(activation_filtered)] = activation_filtered
        current_index += len(activation)
    
    return updated_predicted_energy
