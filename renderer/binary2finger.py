import argparse
import cv2
import math
import numpy as np
import os
import pickle
import random
import sys
import tensorflow as tf
import time
import utils

from numpy import linalg as LA
from scipy.misc import bytescale
from tqdm import tqdm

from nntools.tensorflow.networks import BigGAN
from ..image_utils import pad_image, random_rotate, random_translate

def main(args):
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    random.seed(0)
    NUM_IMPRESSIONS = 1

    # Get the configuration file
    config = utils.import_file(os.path.join(args.model_dir, 'config.py'), 'config')
    
    # Get the paths of the minutiae mpas
    binary_paths = [os.path.join(args.input_dir, bin_path) for bin_path in os.listdir(args.input_dir)]

    # Load model files and config file
    network = BigGAN()
    network.load_model(args.model_dir)
    for bin_image in tqdm(binary_paths):
        binary_image = cv2.imread(bin_image, 0)

        binary_image = pad_image(binary_image, 512, 512)
        binary_image = cv2.resize(binary_image, (256,256))
        
        binary_image = (binary_image - 127.5) / 128.0 
        
        binary_image = np.expand_dims(binary_image, axis=0)
        binary_image = np.expand_dims(binary_image, axis=-1)
        binary_image = np.concatenate([binary_image, binary_image], axis=0)
        
        for i in range(NUM_IMPRESSIONS):
            reconstructed = network.generate_images(binary_image)
            reconstructed = (reconstructed + 1.0) * 127.5

            path = os.path.join(args.output_dir, bin_image.split('/')[-1])

            #random rotation and translations
            out_img = random_rotate(reconstructed[0], 15)
            out_img = random_translate(out_img, (25, 50))

            cv2.imwrite(path, out_img)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str, required=True)
    parser.add_argument("--input_dir", help="Input directory of binary images",
                        type=str, default='example_master_prints_warped')
    parser.add_argument("--output_dir", help="Output directory to save rendered images",
                        type=str, default='example_master_prints_rendered')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=1)
    args = parser.parse_args()
    main(args)
