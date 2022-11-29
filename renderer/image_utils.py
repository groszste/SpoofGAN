import cv2
import os
import sys
import numpy as np
import random
from datetime import datetime
from scipy import ndimage
import math
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
import copy
import scipy.ndimage as nd
import glob
from tqdm import tqdm


def get_new_shape(img, out_w, out_h):
    shape = list(img.shape)
    shape[0] = out_h
    shape[1] = out_w
    shape = tuple(shape)
    return shape

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

def top_center_crop(img, out_w, out_h):
    h,w = img.shape[:2]
    x = w // 2
    x1 = max(0, x - (out_w // 2))
    x2 = min(w, x + (out_w // 2))
    y1, y2 = 0, out_h

    crop = img[y1:y2, x1:x2]
    return crop

def random_crop(img, out_w, out_h):
    _h, _w = img.shape[:2]
    shape_new = get_new_shape(img, out_w, out_h)
    assert (_h>=out_h and _w>=out_w)

    img_new = np.ndarray(shape_new, dtype=img.dtype)

    y = np.random.randint(low=0, high=_h-out_h+1)
    x = np.random.randint(low=0, high=_w-out_w+1)

    img_new = img[y:y+out_h, x:x+out_w]
    return img_new

def crop(img, out_w, out_h):
    h, w = img.shape[:2]
    x = w // 2
    y = h // 2

    x1 = max(0, x - (out_w // 2))
    x2 = min(w, x + (out_w // 2))

    y1 = max(0, y - (out_h // 2))
    y2 = min(h, y + (out_h // 2))

    crop = img[y1:y2, x1:x2]
    return crop

def center_crop(img, out_w, out_h):
    _h, _w = img.shape[:2]
    shape_new = get_new_shape(img, out_w, out_h)
    assert (_h>=out_h and _w>=out_w)

    img_new = np.ndarray(shape_new, dtype=img.dtype)

    y = int(round(0.5 * (_h - out_h)))
    x = int(round(0.5 * (_w - out_w)))

    img_new = img[y:y+out_h, x:x+out_w]
    return img_new, y, x

def flip(img):
    return np.fliplr(img)

def random_flip(img):
    if np.random.rand(1) >= 0.5:
        return flip(img)
    return img

def binarize(img):
    kernel = np.ones((15, 15), np.float32)/225.0
    gradient_img = cv2.Laplacian(img, cv2.CV_64F)
    gradient_img = np.uint8(np.absolute(gradient_img))
    ret, contrast_img = cv2.threshold(gradient_img, 25, 255, cv2.THRESH_BINARY)
    smoothed_img = cv2.filter2D(contrast_img, -1, kernel)   
    ret, binary_img = cv2.threshold(smoothed_img, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_img = cv2.erode(binary_img, kernel, iterations = 2)
    binary_img = cv2.dilate(binary_img, kernel, iterations = 2)
    return binary_img

def segment(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    binary_img = binarize(img)
    
    contours, hierarchy = cv2.findContours(binary_img,cv2.RETR_LIST, \
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx

    # Output to files
    roi=img[y:y+h,x:x+w]
    
    return roi

def center_crop(img, out_w, out_h):
    h,w = img.shape[:2]
    x = w // 2
    y = h // 2
    x1 = x - (out_w // 2)
    x2 = x + (out_w // 2)
    y1 = y - (out_h // 2)
    y2 = y + (out_h // 2)

    crop = img[y1:y2, x1:x2]
    return crop

def resize(img, w, h):
    _h, _w = img.shape[:2]
    shape_new = get_new_shape(img, w, h)
    return imresize(img, (h,w))

def standardize_image(img, standard):
    if standard=='mean_scale':
        mean = 127.5
        std = 128.0
    elif standard=='scale':
        mean = 0.0
        std = 255.0
    return (img.astype(np.float32) - mean) / std

def random_brightness(img, max_delta):
    # randomly adjust the brightness of the image by -max delta to max_delta
    delta = random.uniform(-1.0*max_delta, max_delta)
    return np.clip(img + delta, 0, 255).astype(np.uint8)

def random_rotate(img, degrees):
    # randomly rotate from 0 to degrees and from 0 to negative degrees
    deg = random.uniform(-1.0*degrees, degrees)
    if deg < 0:
      deg = 360 + deg
    return rotate(img, deg, reshape=False, cval=255.0)

def random_translate(image, units):

    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    
    units_x, units_y = units

    x_units = random.randint(-units_x, units_x)
    y_units = random.randint(-units_y, units_y)

    h, w = image.shape[:2]

    translation_matrix = np.float32([[1,0,x_units], [0,1,y_units]])

    translated = cv2.warpAffine(image, translation_matrix, (w, h), borderValue=(255.0,255.0,255.0))

    return cv2.cvtColor(translated, cv2.COLOR_RGB2GRAY)

def random_obfuscation(img, block_size, fraction):
    height, width = img.shape[:2]
    img_new = copy.deepcopy(img)

    if height % block_size != 0 or width % block_size != 0:
        print("block size must evely divide into img width {} and height {}".format(width, height))

    num_blocks = (height // block_size) * (width // block_size) 
    cells = list(range(num_blocks))

    cells = random.sample(cells, int(fraction * num_blocks))
    indices = [0] * num_blocks
    for e in cells: indices[e] = 1

    i = 0
    for r in range(0, height, block_size):
        for c in range(0, width, block_size):
            if indices[i] == 1:
                # set this block to zeros
                if len(img.shape) == 3:
                    img_new[r:r+block_size, c:c+block_size, :] = np.zeros((block_size, block_size, 3)) * 255
                else:
                    img_new[r:r+block_size, c:c+block_size] = np.zeros((block_size, block_size)) * 255
            i += 1

    return img_new

def main():


    # step 1: resize to 416x560 for matlab distortion field code
    d = '../vloss_generation_binary/example_master_prints'
    out_d = 'example_master_prints_416x560'

    if not os.path.isdir(out_d):
        os.makedirs(out_d)

    paths = glob.glob(d + '/*.jpg')

    for path in tqdm(paths):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (512,512))
        img = pad_image(img, 560, 560)
        img = crop(img, 416, 560)
        out_path = path.replace(d,out_d)
        cv2.imwrite(out_path, img)


    #step 2: disort images using matlab code


    # #step 3: pad to 512x512, random rotate and translations, and resize to 256x256
    d = 'example_master_prints_416x560_warped'
    out_d = 'example_master_prints_256x256_warped_no_augs'

    if not os.path.isdir(out_d):
        os.makedirs(out_d)

    paths = [os.path.join(d,f) for f in os.listdir(d) if f.endswith('.jpg')]
    paths.sort()
    for path in tqdm(paths):
        img = cv2.imread(path, 0)
        # img = segment(img)
        img = pad_image(img, 512, 512)
        img = random_rotate(img, 30)
        img = random_translate(img, (50, 100))
        img = cv2.resize(img, (256,256))
        out_path = path.replace(d,out_d)
        cv2.imwrite(out_path, img)


if __name__ == "__main__":
    main()