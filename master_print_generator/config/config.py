''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

# The name of the current model for output
name = 'biggan_z2binary_cm_live'

# The folder to save log and model
log_base_dir = 'log'

# Whether to save the model checkpoints and result logs
save_model = True

# The interval between writing summary
summary_interval = 100

# Prefix to the image files
train_dataset_path = 'z2binary.txt'
# Test data listi
test_dataset_path = 'z2binary.txt'

# Target image size (h,w) for the input of network
image_size = (256,256)

# 3 channels means RGB, 1 channel for grayscale
channels = 1

batch_format='random_samples'

# Preprocess for training
preprocess_train = [
        ['resize', (256, 256)],
        ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    ['standardize', 'mean_scale'],
]

# Number of GPUs
num_gpus = 2


####### NETWORK #######

# The network architecture
network = 'nets/biggan256.py'

# Optimizer
optimizer_d = ("MAO", {'beta1': 0.0, 'beta2': 0.9})
optimizer_g = ("ADAM", {'beta1': 0.0, 'beta2': 0.9})

# Number of samples per batch
batch_size = 8

# Number of batches per epoch
epoch_size = 100

# Number of epochs
num_epochs = 500

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
g_lr = 0.0001
#g_lr = 0.01
g_learning_rate_schedule = {
    0: 1 * g_lr,
}
#d_lr = 0.0004
d_lr = 0.0004
d_learning_rate_schedule = {
    0: 1 * d_lr,
}

learning_rate_multipliers = {
    #'LocalizationNet': 0.00001
    #'LocalizationNet': 1.0,
    #'generator': 0.01,
    #'discriminator': 0.01
}

# Restore model
restore_model = 'log/biggan_z2binary/256x256'

# Keywords to filter restore variables, set None for all
#restore_scopes = ['generator', 'discriminator']
restore_scopes=None

# Weight decay for model variables
weight_decay = 0.0

# Keep probability for dropouts
keep_prob = 1.0

# z dim for style modulation
z_dim = 256

####### LOSS FUNCTION #######

g_loss = 'gan'
g_iters = 2
d_iters = 1

