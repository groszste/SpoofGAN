import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
from functools import partial

import utils
# import visualize
from nntools.common.dataset import Dataset
from nntools.common.imageprocessing import *
from nntools.tensorflow.networks import BigGAN, BaseNetwork
import random

#random.seed(0)

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def main(args):
    config_file = args.config_file
    # I/O
    config = utils.import_file(config_file, 'config')

    trainset = Dataset(config.train_dataset_path)
    testset = Dataset(config.test_dataset_path)
    
    network = BigGAN()
    network.initialize(config)

    # Initalization for running
    log_dir = utils.create_log_dir(config, config_file)
    summary_writer = tf.summary.FileWriter(log_dir, network.graph)
        
    if config.restore_model:
        network.restore_model(config.restore_model, config.restore_scopes)
    proc_func = lambda images, maps: preprocess(images, maps, config, True)
    trainset.start_batch_queue(config.batch_size, batch_format=config.batch_format, proc_func=proc_func)

    #
    # Main Loop
    #
    print('\nStart Training\n# epochs: %d\nepoch_size: %d\nbatch_size: %d\n'\
        % (config.num_epochs, config.epoch_size, config.batch_size))
    global_step = 0
    start_time = time.time()
    for epoch in range(config.num_epochs):
        # Training
        for step in range(config.epoch_size):
            # Prepare input
            learning_rate_g = utils.get_updated_learning_rate(global_step, config)
            learning_rate_d = utils.get_updated_learning_rate(global_step, config, False)
            batch = trainset.pop_batch_queue()
            wl, sm, global_step = network.train(batch['images'],batch['minutiae_labels'], learning_rate_g, learning_rate_d)
            wl['lr'] = learning_rate_g

            # Display
            if step % config.summary_interval == 0:
                duration = time.time() - start_time
                start_time = time.time()
                utils.display_info(epoch, step, duration, wl)
                summary_writer.add_summary(sm, global_step=global_step)
        
        network.save_model(log_dir, global_step)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str)
    args = parser.parse_args()
    main(args)
