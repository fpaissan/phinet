from huggingface_hub import hf_hub_download
from pathlib import Path
import yaml

# timm configuration
aa = "rand-m8-inc1-mstd101"
aug_repeats = 0
aug_splits = 0
batch_size = 256
bce_loss = False
bce_target_thresh = None
bn_eps = None
bn_momentum = None
channels_last = False
checkpoint_hist = 10
class_map = ''
clip_grad = 3.0
clip_mode = 'norm'
color_jitter = 0.4
cooldown_epochs = 10
crop_pct = 1.0
cutmix = 0.0
cutmix_minmax = None
data_dir = "data/cifar10/"
dataset = 'torch/cifar10'
dataset_download = True
decay_epochs = 100
decay_milestones = [30, 60]
decay_rate = 0.1
dist_bn = "reduce"
drop = 0.0
drop_block = None
drop_connect = None
drop_path = 0.1
epoch_repeats = 0.0
epochs = 50
eval_metric = "top1"
experiment = ''
fast_norm = False
fuser = ''
gp = None
grad_checkpointing = False
hflip = 0.5
img_size = None
in_chans = None
initial_checkpoint = ''
interpolation = 'bilinear'
jsd_loss = False
layer_decay = 0.65
local_rank = 0
log_interval = 50
log_wandb = False
lr = 0.001
lr_base = 0.1
lr_base_scale = ''
lr_base_size = 256
lr_cycle_decay = 0.5
lr_cycle_limit = 1
lr_cycle_mul = 1.0
lr_k_decay = 1.0
lr_noise = None
lr_noise_pct = 0.67
lr_noise_std = 1.0
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
min_lr = 5.0e-07
mixup = 0.0
mixup_mode = "batch"
mixup_off_epoch = 0
mixup_prob = 1.0
mixup_switch_prob = 0.5
model = "vit_base_patch32_clip_224.laion2b"
model_ema = True
model_ema_decay = 0.9998
model_ema_force_cpu = False
momentum = 0.9
native_amp = False
no_aug = False
no_ddp_bb = False
no_prefetcher = False
no_resume_opt = False
num_classes = 1000
opt = "adamw"
opt_betas = None
opt_eps = None
output = ''
patience_epochs = 10
pin_mem = False
pretrained = True
ratio = [0.75, 1.3333333333333333]
recount = 1
recovery_interval = 0
remode = "pixel"
reprob = 0.3
resplit = False
resume = ''
save_images = False
scale = [0.08, 1.0]
sched = "cosine"
sched_on_updates = False
seed = 42
smoothing = 0.1
split_bn = False
start_epoch = None
sync_bn = False
torchscript = False
train_interpolation = "random"
train_split = "train"
tta = 0
use_multi_epochs_loader = False
val_split = "validation"
validation_batch_size = None
vflip = 0.0
warmup_epochs = 10
warmup_lr = 1.0e-06
warmup_prefix = False
weight_decay = 0.05
worker_seeding = "all"
workers = 8

# Architecture definition
REPO_ID = "micromind/ImageNet"
FILENAME = "v1/state_dict.pth.tar"

model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
args = Path(FILENAME).parent.joinpath("args.yaml")
args_path = hf_hub_download(repo_id=REPO_ID, filename=str(args))
with open(args_path, "r") as f:
    dat = yaml.safe_load(f)

input_shape = (3, 32, 32)
alpha = dat["alpha"]
num_layers = dat["num_layers"]
beta = dat["beta"]
t_zero = dat["t_zero"]
divisor = 8
downsampling_layers = [5, 7]
return_layers = None
num_classes = 10
