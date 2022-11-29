''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

material = 'all_PAs'
# The name of the current model for output
name = f'biggan_binary2finger_{material}'

# The folder to save log and model
log_base_dir = 'code/biggan/vloss_generation_binary2fing/log'

# Whether to save the model checkpoints and result logs
save_model = True

# The interval between writing summary
summary_interval = 100

# Prefix to the image files
train_dataset_path = f'code/biggan/vloss_generation_binary2fing/labels_{material}.txt'
# Test data listi
test_dataset_path = f'code/biggan/vloss_generation_binary2fing/labels_{material}.txt'

# Target image size (h,w) for the input of network
image_size = (512,512)

# 3 channels means RGB, 1 channel for grayscale
channels = 1

batch_format='random_samples'

# Preprocess for training
preprocess_train = [
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
network = 'code/biggan/vloss_generation_binary2fing/nets/biggan.py'

localization_net = 'code/biggan/vloss_generation_binary2fing/nets/localization_net_tps.py'

# Optimizer
optimizer_d = ("MAO", {'beta1': 0.0, 'beta2': 0.9})
optimizer_g = ("ADAM", {'beta1': 0.0, 'beta2': 0.9})

# Number of samples per batch
batch_size = 4

# Number of batches per epoch
epoch_size = 100

# Number of epochs
num_epochs = 500

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
g_lr = 0.0001
g_learning_rate_schedule = {
    0: 1 * g_lr,
}
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
restore_model = 'code/biggan/vloss_generation_binary2fing/log/biggan_binary2finger_Live/20220320-095717'

# Keywords to filter restore variables, set None for all
restore_scopes = None
# exclude_scopes = ['Moving']
exclude_scopes = None

# Weight decay for model variables
weight_decay = 0.0

# Keep probability for dropouts
keep_prob = 1.0

# z dim for style modulation
z_dim = 128

####### LOSS FUNCTION #######

g_loss = 'gan'
g_iters = 3
d_iters = 1

img_weight = 3.0
dp_t_w = 1.0
dp_m_w = 1.0
# sb_loss_weight = 10.0