# SpoofGAN
- [Introduction](#Introduction)
- [Database](#Database)
- [Inference](#Inference)
- [Models](#Models)
- [Getting Started (Documentation)](#Getting-Started-Documentation)
- [License](#License)
- [Citing](#Citing)

## Introduction

This repository contains the code and a pre-generated dataset associated with the paper "SpoofGAN: Synthetic Fingerprint Spoof Images".

## Database

A preprepared dataset of synthetic fingerprints (live and spoof) is available for download at this link: http://biometrics.cse.msu.edu/Publications/Databases/MSU_SpoofGAN/

## Inference

To generate synthetic images run the generate.py script with the appropriate arguments.

	CUDA_VISIBLE_DEVICES=0 python generate.py --material --output_dir --num_fingers --num_impressions --seed --start --random_ckpt

	Arguments:
		--material: Material type to generate. Default='live'.
		--output_dir: Output save directory. Default='output'.
		--num_fingers: Number of unique finger IDs to generate. Default=10.
		--num_impressions: Number of impressions per finger. Default=3
		--seed: Random Seed. Default=12.
		--start: Starting finger ID. Default=0.
		--random_ckpt: Load random model ckpt rather than latest. Default=True.

Example:

	CUDA_VISIBLE_DEVICES=0 python generate.py --material ecoflex --output_dir output --num_fingers 10 --num_impressions 3

The list of possible materials to choose from is given below:

	live
	pa
	body_double
	conductive_ink
	dragon_skin
	ecoflex
	gelatin
	gold_fingers
	gummy_overlay
	knox_gelatin
	playdoh
	tattoo
	transparency
	woodglue


## Models

All model checkpoints can be found at https://drive.google.com/drive/folders/1ntdAwUJzFuozTqonaoxGlt7NNhCGB6pe?usp=sharing. 

## Getting Started (Documentation)

...

## License

MIT License

Copyright (c) [2022] [Steven A. Grosz]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citing

If you use this code in your work please cite the following paper:

S.A. Grosz and A. K. Jain, "Spoofgan: Synthetic fingerprint spoof images," IEEE Transactions on Information Forensics and Security, 2022.