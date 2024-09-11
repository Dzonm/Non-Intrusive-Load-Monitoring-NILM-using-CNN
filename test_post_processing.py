# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 01:16:31 2024

@author: ID
"""

"""Test a neural network to perform energy disaggregation.

Copyright (c) 2022~2024 Lindo St. Angel
"""

import os
import argparse
import socket

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from logger import Logger
from nilm_metric import NILMTestMetrics
import common
from window_generator import WindowGenerator
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_arguments():
    parser = argparse.ArgumentParser(description='Predict appliance\
        given a trained neural network for energy disaggregation -\
        network input = mains window; network target = the states of\
        the target appliance.')
    parser.add_argument('--appliance_name',
        type=str,
        default='kettle',
        choices=['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine'],
        help='the name of target appliance')
    parser.add_argument('--datadir',
        type=str,
        default='C:/Users/ID/nilmtk_test/mynilm/UK-DALE/Data-set_alt',
        help='this is the directory to the test data')
    parser.add_argument('--trained_model_dir',
        type=str,
        default='C:/Users/ID/nilmtk_test/mynilm/ml_alt/models',
        help='this is the directory to the trained models')
    parser.add_argument('--model_arch',
        type=str,
        default='cnn',
        help='model architecture to test')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='C:/Users/ID/nilmtk_test/mynilm/ml_alt/models',
        help='directory to save test results')
    parser.add_argument('--plot', action='store_true',
        help='if set, plot the predicted appliance against ground truth')
    parser.add_argument('--crop',
        type=int,
        default=None,
        help='use part of the dataset for testing')
    parser.add_argument('--batch_size',
        type=int,
        default=1024,
        help='sets test batch size')
    parser.set_defaults(plot=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    appliance_name = args.appliance_name
    logger = Logger(
        log_file_name = os.path.join(
            args.save_dir, appliance_name, f'{appliance_name}_test_processed_{args.model_arch}.log'
        )
    )
    logger.log(f'Machine name: {socket.gethostname()}')
    logger.log('Arguments: ')
    logger.log(args)

    test_filename = common.find_test_filename(
        args.datadir, appliance_name, 'test'
    )
    logger.log('File for test: ' + test_filename)
    test_file_path = os.path.join(args.datadir, appliance_name, test_filename)
    logger.log('Loading from: ' + test_file_path)

    test_set_x, test_set_y, test_set_y_status = common.load_dataset(
        test_file_path, args.crop
    )
    logger.log(f'There are {test_set_x.size/10**6:.3f}M test samples.')

    window_length = common.params_appliance[appliance_name]['window_length']
    logger.log(f'Window length: {window_length} (samples)')

    sample_period = common.SAMPLE_PERIOD
    logger.log(f'Sample period: {sample_period} (s)')

    max_power = common.params_appliance[appliance_name]['max_on_power']
    logger.log(f'Appliance max power: {max_power} (W)')

    threshold = common.params_appliance[appliance_name]['on_power_threshold']
    logger.log(f'Appliance on threshold: {threshold} (W)')

    train_app_std = common.params_appliance[appliance_name]['train_app_std']
    train_agg_std = common.params_appliance[appliance_name]['train_agg_std']
    test_app_mean = common.params_appliance[appliance_name]['test_app_mean']
    test_agg_mean = common.params_appliance[appliance_name]['test_agg_mean']
    train_agg_mean = common.params_appliance[appliance_name]['train_agg_mean']
    train_app_mean = common.params_appliance[appliance_name]['train_app_mean']
    logger.log(f'Train aggregate mean: {train_agg_mean} (W)')
    logger.log(f'Train aggregate std: {train_agg_std} (W)')
    logger.log(f'Train appliance mean: {train_app_mean} (W)')
    logger.log(f'Train appliance std: {train_app_std} (W)')
    logger.log(f'Test appliance mean: {test_app_mean} (W)')
    logger.log(f'Test aggregate mean: {test_agg_mean} (W)')

    alt_app_mean = common.params_appliance[appliance_name]['alt_app_mean']
    alt_app_std = common.params_appliance[appliance_name]['alt_app_std']
    logger.log(f'Alternative appliance mean: {alt_app_mean} (W)')
    logger.log(f'Alternative appliance std: {alt_app_std} (W)')

    alt_agg_mean = common.ALT_AGGREGATE_MEAN
    alt_agg_std = common.ALT_AGGREGATE_STD
    logger.log(f'Alternative aggregate mean: {alt_agg_mean} (W)')
    logger.log(f'Alternative aggregate std: {alt_agg_std} (W)')

    if common.USE_ALT_STANDARDIZATION:
        logger.log('Using alt standardization.')
    else:
        logger.log('Using default standardization.')

    test_provider = WindowGenerator(
        dataset=(test_set_x, None, None),
        window_length=window_length,
        batch_size=args.batch_size,
        train=False,
        shuffle=False
    )

    def gen():
        """Yields batches of data from test_provider for tf.data.Dataset."""
        for _, batch in enumerate(test_provider):
            yield batch

    test_dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, window_length, 1), dtype=tf.float32)
        )
    )

    # Load best saved trained model for appliance.
    model_file_path = os.path.join(
        args.trained_model_dir, appliance_name, f'savemodel_{args.model_arch}')
    logger.log(f'Loading saved model from {model_file_path}.')
    model = tf.keras.models.load_model(model_file_path, compile=False)

    model.summary()

    test_prediction = model.predict(
        test_dataset,
        steps=len(test_provider),
        verbose=1,
        workers=24,
        use_multiprocessing=True
    )

    # Find ground truth which is center of test target (y) windows.
    # Calculate center sample index of a window.
    center = int(0.5 * (window_length - 1))
    # Calculate ground truth indices.
    ground_truth_indices = np.arange(test_set_y.size - window_length) + center
    # Grab only the center point of each window in target set.
    ground_truth = test_set_y[ground_truth_indices]
    # Adjustment since test_provider defaults to producing complete batches.
    # See `allow_partial_batches' parameter in window_generator.
    ground_truth = ground_truth[:test_prediction.size]

    # De-normalize appliance power predictions.
    if common.USE_APPLIANCE_NORMALIZATION:
        app_mean = 0
        app_std = common.params_appliance[appliance_name]['max_on_power']
    else:
        alt_app_mean = common.params_appliance[appliance_name]['alt_app_mean']
        alt_app_std = common.params_appliance[appliance_name]['alt_app_std']
        app_mean = alt_app_mean if common.USE_ALT_STANDARDIZATION else train_app_mean
        app_std = alt_app_std if common.USE_ALT_STANDARDIZATION else train_app_std
        logger.log('Using alt standardization.' if common.USE_ALT_STANDARDIZATION
                    else 'Using default standardization.')
    logger.log(f'De-normalizing predictions with mean = {app_mean} and std = {app_std}.')
    prediction = test_prediction.flatten() * app_std + app_mean
    # Remove negative energy predictions
    prediction[prediction <= 0.0] = 0.0

    # De-normalize ground truth.
    ground_truth = ground_truth.flatten() * app_std + app_mean

    # De-normalize aggregate data.
    agg_mean = alt_agg_mean if common.USE_ALT_STANDARDIZATION else train_agg_mean
    agg_std = alt_agg_std if common.USE_ALT_STANDARDIZATION else train_agg_std
    aggregate = test_set_x.flatten() * agg_std + agg_mean

    # Calculate ground truth and prediction status.
    prediction_status = np.array(common.compute_status(prediction, appliance_name))
    ground_truth_status = test_set_y_status[ground_truth_indices]
    # Adjustment since test_provider defaults to producing complete batches.
    # See `allow_partial_batches' parameter in window_generator.
    ground_truth_status = ground_truth_status[:test_prediction.size]
    assert prediction_status.size == ground_truth_status.size
    

    
    # After the prediction step:
    
    # Function to compute metrics and log results
    def compute_and_log_metrics(prediction, ground_truth, ground_truth_status, sample_period, logger, prefix=""):
        prediction_status = np.array(common.compute_status(prediction, appliance_name))
        metrics = NILMTestMetrics(
            target=ground_truth,
            target_status=ground_truth_status,
            prediction=prediction,
            prediction_status=prediction_status,
            sample_period=sample_period
        )
        logger.log(f"{prefix} Results:")
        logger.log(f"True positives: {metrics.get_tp()}")
        logger.log(f"True negatives: {metrics.get_tn()}")
        logger.log(f"False positives: {metrics.get_fp()}")
        logger.log(f"False negatives: {metrics.get_fn()}")
        logger.log(f"Accuracy: {metrics.get_accuracy()}")
        logger.log(f"MCC: {metrics.get_mcc()}")
        logger.log(f"F1: {metrics.get_f1()}")
        logger.log(f"MAE: {metrics.get_abs_error()['mean']} (W)")
        logger.log(f"NDE: {metrics.get_nde()}")
        logger.log(f"SAE: {metrics.get_sae()}")
        epd_gt = metrics.get_epd(ground_truth * ground_truth_status, sample_period)
        logger.log(f"Ground truth EPD: {epd_gt} (Wh)")
        epd_pred = metrics.get_epd(prediction * prediction_status, sample_period)
        logger.log(f"Predicted EPD: {epd_pred} (Wh)")
        logger.log(f"EPD Relative Error: {100.0 * (epd_pred - epd_gt) / epd_gt} (%)")
        return prediction, prediction_status, metrics
    
    
    # Apply post-processing
    logger.log("Applying post-processing to predictions...")
    reshaped_predictions = [pred.flatten() for pred in np.split(prediction, len(prediction))]
    reshaped_ground_truth = [gt.flatten() for gt in np.split(ground_truth, len(ground_truth))]
    processed_prediction = common.post_process_activations(reshaped_ground_truth, reshaped_predictions,appliance_name)
    
    # Compute metrics for processed predictions
    processed_prediction, processed_status, processed_metrics = compute_and_log_metrics(
        processed_prediction, ground_truth, ground_truth_status, sample_period, logger, prefix="Processed"
    )
    
    # Remove file extension for saving and plotting results.
    test_filename = os.path.splitext(test_filename)[0]

    # Save results
    save_path = os.path.join(args.save_dir, appliance_name)
    logger.log(f'Saving ground truth, and processed_prediction to {save_path}.')
    np.save(f'{save_path}/{test_filename}_pred_{args.model_arch}_processed.npy', processed_prediction)
    np.save(f'{save_path}/{test_filename}_gt.npy', ground_truth)

    
    
    
    # Plot results
    if args.plot:
        pad = np.zeros(center)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        #ax1.plot(aggregate, color='#7f7f7f', linewidth=1.8)
        ax1.plot(np.concatenate((pad, ground_truth, pad), axis=None),
                 color='#d62728', linewidth=1.8)
        ax1.plot(np.concatenate((pad, processed_prediction, pad), axis=None),
                 color='#1f77b4', linestyle='dotted', linewidth=1.7)
        ax1.grid()
        ax1.set_title('Power Test results on {:}'
            .format(test_filename), fontsize=16, fontweight='bold', y=1.08)
        ax1.set_ylabel('Power ')
        ax1.legend(['ground truth', 'prediction'])
        
        # Plot appliance status
        ax2.plot(np.concatenate((pad, ground_truth_status, pad), axis=None) * max_power,
                 color='#2ca02c', linewidth=1.8)
        ax2.plot(np.concatenate((pad, processed_status, pad), axis=None) * max_power,
                 color='#ff7f0e', linestyle='dotted', linewidth=1.7)
        ax2.grid()
        ax2.set_title('status Test results on {:}'
            .format(test_filename), fontsize=16, fontweight='bold', y=1.08)
        ax2.set_ylabel('Power ')
        ax2.legend(['ground truth on status', 'prediction on status'])

        plot_name = f'processed Test results on {test_filename}'
        plot_filepath = os.path.join(args.save_dir, appliance_name, f'{plot_name}')
        logger.log(f'Plot directory: {plot_filepath}')
        plt.savefig(fname=plot_filepath)
        
        plt.show()
        plt.close()