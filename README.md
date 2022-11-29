# SpoofGAN
- [Introduction](#Introduction)
- [Inference](#Inference)
- [Training](#Training)
- [Models](#Models)
- [Getting Started (Documentation)](#Getting-Started-Documentation)
- [Licenses](#Licenses)
- [Citing](#Citing)

## Introduction

## Inference

### Master-print-generation
* CUDA_VISIBLE_DEVICES=0 python master_print_generator/z2binary.py --model_dir master_print_generator/log/biggan_z2binary_cm_live --output_dir example_master_prints --num_fingers 10 --batch_size 1

### Statistical-warping
* python STN/warp_images.py --num_impressions 3 --input_dir example_master_prints --output_dir example_master_prints_warped

### Texture-rendering
* CUDA_VISIBLE_DEVICES=0 python renderer/binary2finger.py --model_dir renderer/log/biggan_binary2finger_Live/final --input_dir example_master_prints_warped --output_dir example_master_prints_rendered

## Training
* to be added later

### Master-print-generation
* ...

### Statistical-warping
* ...

### Texture-rendering
* ...

## Models

All model checkpoints can be found at https://drive.google.com/drive/folders/1ntdAwUJzFuozTqonaoxGlt7NNhCGB6pe?usp=sharing. 

## Getting Started (Documentation)

...

## Licenses

The code and models here are licensed Apache 2.0.

## Citing

Please cite the following paper...

