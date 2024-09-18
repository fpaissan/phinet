"""
Configuration file image classification with PhiNet.

Authors:
    - Francesco Paissan, 2023
"""

import torchlibrosa


def get_model_from_str(s, vs=("alpha", "beta", "t0", "N")):
    def get_var(s, key):
        tmp = s.split("_")
        return tmp[tmp.index(key) + 1]

    verb = "PhiNet initialized with "
    ret = {}
    for k in vs:
        verb += f"{k}={get_var(s, k)} "
        ret[k] = float(get_var(s, k))

    ret["t_zero"] = ret["t0"]
    ret["num_layers"] = ret["N"]
    del ret["t0"]
    del ret["N"]

    return ret


name = "phinet_alpha_1.50_beta_0.75_t0_6_N_7"

phinet_conf = get_model_from_str(name)

esc50_folder = "ESC-50"
sample_rate = 44100

# pre-processing
n_mels = 64
spec_mag_power = 1
fmin = 50
fmax = 14000
n_fft = 1024
hop_length = 320
win_length = 1024
use_melspectra_log1p = False
use_melspectra = True
use_stft2mel = True

# Spectrogram extractor
spectrogram_extractor = torchlibrosa.stft.Spectrogram(
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window="hann",
    center=True,
    pad_mode="reflect",
    freeze_parameters=True,
)

# Logmel feature extractor
logmel_extractor = torchlibrosa.stft.LogmelFilterBank(
    sr=sample_rate,
    n_fft=win_length,
    n_mels=n_mels,
    fmin=fmin,
    fmax=fmax,
    ref=1.0,
    amin=0.0000000001,
    top_db=None,
    freeze_parameters=True,
)

# Model configuration
model = "phinet"
input_shape = (3, 32, 32)
alpha = 3
num_layers = 7
beta = 1
t_zero = 5
divisor = 8
downsampling_layers = [5, 7]
return_layers = None

ckpt_pretrained = ""

# Basic training loop
epochs = 50
lr = 1e-6

# Basic data
data_dir = "data/cifar10/"
dataset = "torch/cifar10"
batch_size = 16
dataset_download = True

# Dataloading config
num_workers = 4
pin_memory = True
persistent_workers = True
