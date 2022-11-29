import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import glob
from model import CycleGAN
from config import *
from utils import to_data
import sys

test_path = sys.argv[1]+"/*"
save_path = sys.argv[2]
quality = sys.argv[3]

G_XtoY,_,_,_,_,_ = CycleGAN(n_res_blocks=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def translateDomain(img_x, model):
    img_x = cv2.resize(img_x, image_size)
    img_x = np.asarray((np.true_divide(img_x, 255.0) - 0.5), dtype=np.float32)
    torch_img_x = np.moveaxis(img_x, -1, 0)
    torch_img_x = torch_img_x[np.newaxis, ...]
    img_x_tensor = torch.tensor(torch_img_x)
    img_x_tensor = img_x_tensor.to(device)
    model.eval()
    torch_img_y = model(img_x_tensor)
    img_y = to_data(torch_img_y[0])
    img_y = np.moveaxis(img_y, 0, 2)

    return img_y

if __name__ == '__main__':

    G_XtoY.load_state_dict(torch.load("models/"+quality+"/G_XtoY.pkl", map_location=lambda storage, loc: storage))
    print('Loaded pretrained weights')

    img_fnames = glob.glob(test_path)

    for fname in img_fnames:
        img_x_original = cv2.imread(fname)
        img_x = img_x_original
        img_orig_size = (img_x_original.shape[1], img_x_original.shape[0])
        img_x_rgb = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        img_y = translateDomain(img_x_rgb, G_XtoY)
        img_y_bgr = cv2.cvtColor(img_y, cv2.COLOR_RGB2BGR)
        img_y_bgr = cv2.resize(img_y_bgr, img_orig_size, interpolation=cv2.INTER_LINEAR)
        file = fname.split("/")
        print(save_path+file[len(file)-1])
        cv2.imwrite(save_path+file[len(file)-1], img_y_bgr)

