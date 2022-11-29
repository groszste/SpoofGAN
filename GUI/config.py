## hyper parameters
p_lr=0.0003                     # Learning Rate of Generator
g_lr=0.0003                     # Learning Rate of Generator
d_lr=0.0001                     # Learning Rate of Discriminator
beta1=0.5                       # Beta 1 for Adam optimizer
beta2=0.999                     # Beta 2 for Adam optimizer
    
batch_size=2                    # batch size
init_epoch=0                    # initial epoch
n_epochs=500                    # maximum nomber of epochs
n_samples=-1                    # -1 for all available samples
test_size=0.1                  # fraction of all samples for validation
early_stop_epoch_thres=50       # threshold for stopping training if loss does not improve
    
image_size = (512, 512)         # image size 

# lambda for losses
lambda_discriminator=1.0        # lambda weight for discriminator loss
lambda_cycle_consistency=10.0   # lambda weight for cycle-consistency loss

# flags
use_pretrained_weights=True

# experiment id
experiment_id = 'M'

## paths
domain_a_dir = '/scratch0/Andre_2/August/New_cycleganmodel3/SD4_first_flip/*.jpg'
domain_b_dir = '/scratch0/Andre_2/August/data/Msp_latent_center_mirror_9_clusters/6/*.jpg'

# directories
checkpoints_dir = 'checkpoints'
samples_dir = 'samples'
logs_dir = 'logs'

# pretrained weights
generator_x_y_weights = './checkpoints/'+experiment_id+'/best/G_XtoY.pkl'
generator_y_x_weights = './checkpoints/'+experiment_id+'/best/G_YtoX.pkl'
discriminator_xp_weights = './checkpoints/'+experiment_id+'/best/Dp_X.pkl'
discriminator_yp_weights = './checkpoints/'+experiment_id+'/best/Dp_Y.pkl'
discriminator_xg_weights = './checkpoints/'+experiment_id+'/best/Dg_X.pkl'
discriminator_yg_weights = './checkpoints/'+experiment_id+'/best/Dg_Y.pkl'

## Number of epochs to sample and checkpoint
sample_every=1
checkpoint_every=1