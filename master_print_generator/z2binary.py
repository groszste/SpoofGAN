import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
from nntools.tensorflow.networks import BigGAN
import utils
import time

from image_utils import pad_image, crop

def main(args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    RESIZE = True

    np.random.seed(12)

    NUM_FINGERS = args.num_fingers
    start_idx = 0
    BS = args.batch_size

    # Get the configuration file
    config = utils.import_file(os.path.join(args.model_dir, 'config.py'), 'config')
    
    # Load model files and config file
    network = BigGAN()
    network.load_model(args.model_dir)
    curr = 0
    start = time.time()
    for i in range(start_idx, NUM_FINGERS+1, BS):
        print(i, NUM_FINGERS / BS)
        z_noise = np.random.normal(size=(BS, config.z_dim))
        reconstructed = network.generate_images(z_noise)
        reconstructed = (reconstructed + 1.0) * 127.5
        # write out the images to the disk
        for j in range(BS):
            path = os.path.join(args.output_dir, '{}.jpg'.format(curr-1))

            if RESIZE:  # warping code expects inputs of 416x560
                out_img = cv2.resize(reconstructed[j], (512,512))
                out_img = pad_image(out_img, 560, 560)
                out_img = crop(out_img, 416, 560)
            else:
                out_img = reconstructed[j]

            if curr - 1 > -1:
                cv2.imwrite(path, out_img)
            curr += 1
    end = time.time()
    print('duration ', end - start)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str, required=True)
    parser.add_argument("--output_dir", help="Output directory to save master prints",
                        type=str, default='example_master_prints')
    parser.add_argument("--num_fingers", help="Number of fingers to generate",
                        type=int, default=10)
    parser.add_argument("--batch_size", help="Number of images per batch",
                        type=int, default=1)
    args = parser.parse_args()
    main(args)
