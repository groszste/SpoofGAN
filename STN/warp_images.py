import argparse
import cv2
from functools import partial
import glob
import imp
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

# sys.path.append('STN/')
from scipy.io import loadmat
from tps_stn import ElasticTransformer

def pad_image(img, output_w, output_h):
    h, w = img.shape[:2]
    margin_h = abs((output_h - h) // 2)
    margin_w = abs((output_w - w) // 2)

    if len(img.shape) == 3:
        output_img = np.ones((output_h, output_w, 3), dtype=np.uint8) * 255
        if output_h < h and output_w < w:
            output_img = img[margin_h:margin_h+output_h, margin_w:margin_w+output_w, :]
        elif output_h < h:
            output_img[:,margin_w:margin_w+w, :] = img[margin_h:margin_h+output_h, :, :]
        elif output_w < w:
            output_img[margin_h:margin_h+h,:, :] = img[:, margin_w:margin_w+output_w, :]
        else:
            output_img[margin_h:margin_h+h, margin_w:margin_w+w, :] = img
    else:
        output_img = np.ones((output_h, output_w), dtype=np.uint8) * 255
        if output_h < h and output_w < w:
            output_img = img[margin_h:margin_h+output_h, margin_w:margin_w+output_w]
        elif output_h < h:
            output_img[:,margin_w:margin_w+w] = img[margin_h:margin_h+output_h, :]
        elif output_w < w:
            output_img[margin_h:margin_h+h,:] = img[:, margin_w:margin_w+output_w]
        else:
            output_img[margin_h:margin_h+h, margin_w:margin_w+w] = img
    return output_img

def inference(stl, image, x_displacements, y_displacements, n0=9):
    x = x_displacements.reshape((11,9))
    y = y_displacements.reshape((11,9))

    x = cv2.resize(x, dsize=(n0, n0), interpolation=cv2.INTER_CUBIC)
    y = cv2.resize(y, dsize=(n0, n0), interpolation=cv2.INTER_CUBIC)

    x = x / 416
    y = y / 560

    x = x.flatten().tolist()
    y = y.flatten().tolist()
    
    displacements = tf.concat((x, y), axis=0)
    warped_img = stl.transform(tf.convert_to_tensor(image), displacements)
    return warped_img, displacements

def main(args):
    RESIZE = False  # set to True if input size is not (416,512)
    data = loadmat('STN/PCAParameterForDistortedVideo.mat')
    out_size=(560,416)
    n0=4
    mu = 0
    sigma = .66
    
    num_impressions = 3
    num_impressions = args.num_impressions
    f_mean = data['f_mean'].squeeze()
    latent = data['latent'].squeeze()
    coeff = data['coeff']

    img_dir = 'example_master_prints'
    warped_dir = 'example_master_prints_warped'

    stl = ElasticTransformer(out_size, param_dim=2*n0**2)
    sess = tf.Session()
    with sess.as_default():
        paths = glob.glob(img_dir + '/*.jpg')
        paths.sort()
        for i, path in enumerate(paths):
            print(f'processing {i+1} of {len(paths)} fingers')
            img = cv2.imread(path)
            if RESIZE:
                img = cv2.resize(img, (512, 512))
                img = pad_image(img, 416, 560)
            img = np.expand_dims(img, axis=0).astype(np.float32)

            for idx in range(num_impressions):
                start = time.time()
                r1 = np.random.normal(mu, sigma)
                r2 = np.random.normal(mu, sigma)
                d = f_mean + r1*coeff[:, 0]*np.sqrt(latent[0]) + r2*coeff[:, 1]*np.sqrt(latent[1])
                
                dx = d[0::2].astype(np.float32)
                dy = d[1::2].astype(np.float32)

                warped_img, displacements = inference(stl, img, -1*dy, 1*dx, n0=n0)
                warped_img = warped_img.eval()     

                outpath = path.replace(img_dir, warped_dir).replace('.jpg', f'_{idx}.jpg')   

                outd = '/'.join(outpath.split('/')[:-1])
                if not os.path.isdir(outd):
                    os.makedirs(outd)

                cv2.imwrite(outpath, warped_img[0].astype(np.uint8))
                end = time.time()
                print(f'impression {idx+1} out of {num_impressions}... time/image: {end-start}')

                # cv2.imshow('warped', warped_img[0].astype(np.uint8))
                # cv2.imshow('orig', img[0].astype(np.uint8))
                # x = cv2.waitKey(0)
                # if x == ord('e'):
                #     raise Exception('exiting')


class TPSwarp:
    def __init__(self, n0=4):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
        self.n0 = n0
            
    def initialize(self):
        with self.graph.as_default():
            with self.sess.as_default():
                self.inputs = tf.placeholder(tf.float32, shape=[1, 560, 416, 1], name='inputs')
                self.displacements = tf.placeholder(tf.float32, shape=[1, 2*self.n0**2], name='displacements')
                
                self.load_tps()

                warped_images, _ = self.tps(image=self.inputs, displacements=self.displacements)
                self.warped_images = tf.identity(warped_images, name='outputs')

                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())

    def load_tps(self):
        print("Loading TPS module!!!")
        warp_model = imp.load_source('network_model', 'STN/tps_stn.py')
        self.tps = partial(warp_model.inference)

    def warp_images(self, images, displacements):
        feed_dict = {
                    self.inputs: images,
                    self.displacements: displacements,
        }
        result = self.sess.run(self.warped_images, feed_dict=feed_dict)
        return result

    def load_model(self, *args, **kwargs):
        self.inputs = self.graph.get_tensor_by_name('inputs:0')
        self.displacements = self.graph.get_tensor_by_name('displacements:0')

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_impressions", help="Number of impressions to generate per finger",
                        type=int, default=3)
    parser.add_argument("--input_dir", help="Input directory of master prints images",
                        type=str, default='example_master_prints')
    parser.add_argument("--output_dir", help="Output directory to save warped images",
                        type=str, default='example_master_prints_warped')
    args = parser.parse_args()
    main(args)