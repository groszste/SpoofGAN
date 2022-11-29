import PySimpleGUI as sg
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

def run(test_path, quality, save_path):
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

print("Loading: Torch device")
G_XtoY,_,_,_,_,_ = CycleGAN(n_res_blocks=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lst = ["Good", "Bad", "Ugly"]
Status = ['A','B','C','D']

layout = [
    [sg.Text("Michigan State University")],
    [sg.Text("Pattern Recognition and Image Processing (PRIP) Lab")],
    [
    sg.Image(size=(200, 200),filename="msu.png", key="-IMAGE-")],
    [
        sg.Text("Rolled fingerprint folder:"),
        sg.In("NIST_SD04_EXAMPLES", size=(25, 1), enable_events=True, key="-FOLDER_ROLLED-" ),
        sg.FolderBrowse(),
    ],
     [
        sg.Text("Latent fingerprint output folder:"),
        sg.In("output", size=(25, 1), enable_events=True, key="-FOLDER_LATENT-"),
        sg.FolderBrowse(),
    ],
    [sg.Text('Latent quality:', size=(18, 1)), sg.Combo(lst, key='combo', size=(20, 1), enable_events=True, default_value='Good')],
    [sg.Button("Create Latents")]]

# Create the window
window = sg.Window("Synthetic Latent Fingerprint Generator (SLP)", layout, margins=(100, 100), element_justification="center")

folder_rolled = "NIST_SD04_EXAMPLES"
folder_latents = "output"
quality = "Good"
# Create an event loop
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break

    if event == 'combo':    # the event's name is the element's key
        quality = values["combo"]

    if event == "Create Latents"  or event == sg.WIN_CLOSED:
        if folder_rolled == "" or folder_latents == "" or quality == "":
            sg.popup_error(f'Please fill out all fields.')
        else:
            print(folder_rolled)
            print(folder_latents)
            print(quality)
            run(folder_rolled+"/*", quality, folder_latents+"/")
            sg.popup("Finished!")

    if event == "-FOLDER_ROLLED-":
        folder_rolled = values["-FOLDER_ROLLED-"]

    if event == "-FOLDER_LATENT-":
        folder_latents = values["-FOLDER_LATENT-"]


window.close()