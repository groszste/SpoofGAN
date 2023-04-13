#z2binary imports
import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
from master_print_generator.nntools.tensorflow.networks import BigGAN
import utils
import time
import sys
import shutil

sys.path.append('master_print_generator/')
sys.path.append('renderer/')
sys.path.append('STN/')

#warp_images imports
from STN.tps_stn import ElasticTransformer
from STN.warp_images import inference, TPSwarp
import glob
from scipy.io import loadmat

#binary2finger imports
import math
import pickle
from numpy import linalg as LA
from renderer.nntools.tensorflow.networks import BigGAN as BigGANRenderer
from tqdm import tqdm
from scipy.misc import bytescale
import random
from image_utils import random_rotate, random_translate, pad_image


def main(args):

    materials_dict = {
        'live': 'renderer/log/biggan_binary2finger_Live/20220320-100630',
        'pa': 'renderer/log/biggan_binary2finger_all_PAs/20220320-162834',
        'body_double': 'renderer/log/biggan_binary2finger_BodyDouble/20220320-223430',
        'conductive_ink': 'renderer/log/biggan_binary2finger_conductiveinkonpaper/20220323-192639',
        'ecoflex': 'renderer/log/biggan_binary2finger_Ecoflex/20220320-223348',
        'gelatin': 'renderer/log/biggan_binary2finger_Gelatin/20220320-223522',
        'gummy_overlay': 'renderer/log/biggan_binary2finger_Gummy_Overlay/20220323-192604',
        'playdoh': 'renderer/log/biggan_binary2finger_PlayDoh/20220320-223547',
        'tattoo': 'renderer/log/biggan_binary2finger_Tattoo/20220323-192606',
    }

    seed = args.seed
    num_fingers = args.num_fingers
    num_impressions = args.num_impressions
    output_dir = args.output_dir
    random_ckpt = args.random_ckpt
    start_idx = args.start
    material = args.material.lower()
    bs = 1

    output_dir = os.path.join(output_dir, material)

    if material not in materials_dict:
        raise Exception('invalid material')
    else:
        TEXTURE_MODEL_DIR = materials_dict[material]

    if seed < 0:
        seed = random.randint(1, 100)
    np.random.seed(seed)

    rng1 = np.random.RandomState(random.randint(101, 200))
    rng2 = np.random.RandomState(random.randint(101, 200))
    rng3 = np.random.RandomState(random.randint(101, 200))
    
    # Get the configuration file
    mp_model_dir = 'master_print_generator/log/biggan_z2binary_cm_live'
    config = utils.import_file(os.path.join(mp_model_dir, 'config.py'), 'config')
    
    # Load model files and config file
    print('\nloading master print generator....\n')
    network = BigGAN()
    network.load_model(mp_model_dir) 

    print('\nbegin master print generation\n')
    try:
        tempd = 'tempd_' + time.strftime("%Y%m%d-%H%M%S")
        if os.path.isdir(tempd):
            shutil.rmtree(tempd)

        os.makedirs(tempd)

        curr = -1
        start = time.time()
        for i in tqdm(range(0, start_idx+num_fingers+1, bs)):
            z_noise = np.random.normal(size=(bs, config.z_dim))
            reconstructed = network.generate_images(z_noise)
            reconstructed = (reconstructed + 1.0) * 127.5
            # write out the images to the disk
            for j in range(bs):
                if curr >= start_idx:
                    outpath = os.path.join(tempd, str(curr)+'.jpg')
                    out_img = reconstructed[j]
                    cv2.imwrite(outpath, out_img)
                curr += 1

        end = time.time()
        print('\nmaster prints generated, duration: ', end - start)

        #WARP_IMAGES
        data = loadmat('STN/PCAParameterForDistortedVideo.mat')
        out_size=(560,416)
        n0=4
        mu = 0
        # sigma = .66
        sigma = 1.0
        f_mean = data['f_mean'].squeeze()
        latent = data['latent'].squeeze()
        coeff = data['coeff']

        warper = TPSwarp(n0)
        warper.initialize()

        print('\nbegin warping......\n')
        start = time.time()
        img_list = glob.glob(tempd + '/*.jpg')
        for path in tqdm(img_list):
            i = path.split('/')[-1].split('.')[0]

            img = cv2.imread(path, 0)
            img = cv2.resize(img, (512, 512))
            img = pad_image(img, 416, 560)

            img = np.expand_dims(img, axis=0).astype(np.float32)
            img = np.expand_dims(img, axis=-1)

            for idx in range(num_impressions):
                r1 = np.random.normal(mu, sigma)
                r2 = np.random.normal(mu, sigma)
                d = f_mean + r1*coeff[:, 0]*np.sqrt(latent[0]) + r2*coeff[:, 1]*np.sqrt(latent[1])
                
                y_displacements = d[0::2].astype(np.float32)
                x_displacements = d[1::2].astype(np.float32)

                x = x_displacements.reshape((11,9))
                y = y_displacements.reshape((11,9))

                x = cv2.resize(x, dsize=(n0, n0), interpolation=cv2.INTER_CUBIC)
                y = cv2.resize(y, dsize=(n0, n0), interpolation=cv2.INTER_CUBIC)

                x = x / 416
                y = y / 560

                x = x.flatten().tolist()
                y = y.flatten().tolist()
                
                displacements = np.concatenate((x, y), axis=0)

                displacements = displacements.reshape((1,2*n0**2))


                warped_img = warper.warp_images(img, displacements)

                warped_img = warped_img[0].squeeze(axis=-1)

                outpath = os.path.join(tempd, f'{i}_{idx}.jpg')
                cv2.imwrite(outpath, warped_img.astype(np.uint8))
            os.remove(path)

        end = time.time()
        print(f'\nwarping impressions completed, duration: {end-start}')

        #BINARY2FINGER
        random.seed(0)

        # Get the configuration file
        config = utils.import_file(os.path.join(TEXTURE_MODEL_DIR, 'config.py'), 'config')
        
        # Load model files and config file
        print("\nloading renderer model.........\n")
        network = BigGANRenderer()
        network.load_model(TEXTURE_MODEL_DIR, scope=None, random_ckpt=random_ckpt, rng=rng3)

        start = time.time()
        img_list = glob.glob(tempd + '/*.jpg')
        img_list.sort()
        print('\nbegin rendering....\n')
        for path in tqdm(img_list):
            binary_image = cv2.imread(path, 0)
            fname = path.split('/')[-1]

            binary_image = pad_image(binary_image, 512, 512)
            binary_image = cv2.resize(binary_image, (256,256))
            
            binary_image = (binary_image - 127.5) / 128.0 
            
            binary_image = np.expand_dims(binary_image, axis=0)
            binary_image = np.expand_dims(binary_image, axis=-1)
            binary_image = np.concatenate([binary_image, binary_image], axis=0)
            
            reconstructed = network.generate_images(binary_image)
            reconstructed = (reconstructed + 1.0) * 127.5

            ID = fname.split('_')[0]
            path = os.path.join(output_dir, ID, fname)
            outdir = '/'.join(path.split('/')[:-1])
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            #random rotation and translations
            out_img = random_rotate(reconstructed[0], 15, rng=rng1)
            out_img = random_translate(out_img, (100, 50), rng1=rng1, rng2=rng2)

            cv2.imwrite(path, out_img)

        end = time.time()
        print(f'\nrendering completed, duration: {end-start}')

        if os.path.isdir(tempd):
            shutil.rmtree(tempd)  
    except:
        print('\ngeneration failed, exiting.......\n')
        if os.path.isdir(tempd):
            shutil.rmtree(tempd) 

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", help="Material type to generate",
                        type=str, default='live')
    parser.add_argument("--output_dir", help="Output directory to save rendered images",
                        type=str, default='output')
    parser.add_argument("--num_fingers", help="Number of fingers to generate",
                        type=int, default=10)
    parser.add_argument("--num_impressions", help="Number of impressions to generate per finger",
                        type=int, default=3)
    parser.add_argument("--seed", help="Random seed.",
                        type=int, default=12)
    parser.add_argument("--start", help="Starting finger ID",
                        type=int, default=1)
    parser.add_argument('--random_ckpt', action='store_true', help='Load a random ckpt. Using multiple ckpts increases diversity in generated fingerprints')
    args = parser.parse_args()
    main(args)