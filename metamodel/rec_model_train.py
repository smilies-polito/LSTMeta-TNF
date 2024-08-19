import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('--tumor_radius', type=int, default=100, help='Tumor radius')
parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs')
parser.add_argument('--n_layers_in', type=int, default=2, help='Number of layers')
parser.add_argument('--latent_dim', type=int, default=500, help='Latent dimension')
parser.add_argument('--dropout_lin', type=float, default=0.4, help='Linear dropout')
parser.add_argument('--dropout_rec', type=float, default=0.2, help='Recurrent dropout')
parser.add_argument('--window_size', type=int, default=24, help='Length of the window in number of steps')
parser.add_argument('--subsample_every', type=int, default=1, help='Integer interval at which to subsample cell states')
parser.add_argument('--augment_data', type=int, default=2, help='Integer level at which to augment data (0, 1, 2)')

args = parser.parse_args()

ARCH = 'lstm' # 'rnn' or 'lstm'

AUGMENT_DATA = args.augment_data

SINGLE_EXP = True

TUMOR_RADIUS = args.tumor_radius # 50, 100, 275, 400, 'all'

SUBSAMPLE_EVERY = args.subsample_every

WANDB_LOG = True

SEED = 1

TIMESTEP = 30*SUBSAMPLE_EVERY


# ##### Imports

# In[ ]:


import torch
import numpy as np
import random
import time
import wandb

import pandas as pd
import ast
import matplotlib.pyplot as plt

import platform
import os
import socket


# In[ ]:


if WANDB_LOG:
    wandb.login(key="f528cb95c325b980ef950c1fec50ea9707fad591")


# ##### Parameters

# In[ ]:


EXP_DIR = os.path.join("experiments", "radius_new_time")
os.makedirs(EXP_DIR, exist_ok=True)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
device


# In[ ]:


torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ##### Load Data

# In[ ]:


if 'mac' in platform.platform():
    DATA_DIR = "/Users/marco/phd/meta-model-physiboss/metamodel/data/"
    DATA_RADII_DIR = "/Users/marco/phd/meta-model-physiboss/metamodel/data_radii/"
    DATA_RADIUS100_DIR = "/Users/marco/phd/meta-model-physiboss/metamodel/data_radius100_50k/"
elif 'mustafar' in socket.gethostname() or 'daredevil' in socket.gethostname():
    DATA_DIR = "./data/"
    DATA_RADII_DIR = "./data_radii/"
    DATA_RADIUS100_DIR = "./data_radius100_50k/"
else:
   raise ValueError("no known data directory")


# In[ ]:


data = pd.read_csv(os.path.join(DATA_DIR, "data.csv"))

if AUGMENT_DATA >= 1:
    data_radii = pd.read_csv(os.path.join(DATA_RADII_DIR, "data.csv"))

    data_radii['index'] += (data['index'].max()+1)
    data = data.set_index("index")
    data_radii = data_radii.set_index("index")

    data_radii['index_input_params'] += (data["index_input_params"].max()+1)

    data = pd.concat([data, data_radii])
else:
    data = data.set_index("index")

if AUGMENT_DATA >= 2:
    data_radius100 = pd.read_csv(os.path.join(DATA_RADIUS100_DIR, "data.csv"))
    data_radius100['index_input_params'] += (data["index_input_params"].max()+1)
    data_radius100 = data_radius100.set_index("index")


# ##### Prepare Data 

# In[ ]:


CELL_STATES_ORDER = ["alive", "apoptotic", "necrotic"]

if TUMOR_RADIUS != 'all':
    if TUMOR_RADIUS not in data["tumor_radius"].unique():
        raise ValueError(f"Tumor radius {TUMOR_RADIUS} not in data")
    
    print(f"Filtering data for tumor radius {TUMOR_RADIUS}")
    data = data[data["tumor_radius"] == TUMOR_RADIUS]
    INPUT_PARAMS_ORDER = ["time_add_tnf", "duration_add_tnf", "concentration_tnf"]
else:
    INPUT_PARAMS_ORDER = ["time_add_tnf", "duration_add_tnf", "concentration_tnf", "tumor_radius"]


# In[ ]:

if TUMOR_RADIUS == 100 and AUGMENT_DATA >= 2:
    data_radius100['tumor_radius'] = 100
    data_radius100['index_input_params'] += (data["index_input_params"].max()+1)
    data = pd.concat([data, data_radius100])

data = data[data['concentration_tnf'] > 0.1]

# Split train and test
index_input_params = data["index_input_params"].unique()

train_size = int(0.88 * len(index_input_params))
idx_train = np.random.choice(index_input_params, size=train_size, replace=False)
idx_test = np.setdiff1d(index_input_params, idx_train)


# In[ ]:


data[CELL_STATES_ORDER] = data[CELL_STATES_ORDER].map(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)[::SUBSAMPLE_EVERY])

if SINGLE_EXP:
    data = data\
        .groupby("index_input_params")\
        .apply(pd.DataFrame.sample, n=1, random_state=SEED)\
        .reset_index(drop=True).copy()

data_train = data[data["index_input_params"].isin(idx_train)]
data_test = data[data["index_input_params"].isin(idx_test)]

print(len(data_train), len(data_test))


# In[ ]:


WINDOW_SIZE = args.window_size
# BATCH_SIZE_TRAIN = 112 if SINGLE_EXP else 900

if TUMOR_RADIUS == 'all':
    BATCH_SIZE_TRAIN = 438
    BATCH_SIZE_TEST = 59
elif AUGMENT_DATA == 0:
    BATCH_SIZE_TRAIN = 232
    BATCH_SIZE_TEST = 48
elif AUGMENT_DATA == 1:
    BATCH_SIZE_TRAIN = 229
    BATCH_SIZE_TEST = 73
elif AUGMENT_DATA == 2:
    BATCH_SIZE_TRAIN = 268
    BATCH_SIZE_TEST = 64
else:
    BATCH_SIZE_TRAIN = 197
    BATCH_SIZE_TEST = 37

# INPUT_PARAMS_NORM_FACTOR = np.array([705, 165, 1], dtype=np.float32)
INPUT_PARAMS_NORM_FACTOR = np.array([1, 1, 1], dtype=np.float32) #data_train[INPUT_PARAMS_ORDER].max().tolist()

# CELL_STATES_NORM_FACTOR = np.array([230, 105, 110], dtype=np.float32)
CELL_STATES_NORM_FACTOR =  data_train[CELL_STATES_ORDER].map(max).max().tolist()

print("Input parameters normalization factors:")
for idx in range(len(INPUT_PARAMS_ORDER)):
    print(f"\t{INPUT_PARAMS_ORDER[idx]}: {INPUT_PARAMS_NORM_FACTOR[idx]}")
print()
print("Cell states normalization factors:")
for idx in range(len(CELL_STATES_ORDER)):
    print(f"\t{CELL_STATES_ORDER[idx]}: {CELL_STATES_NORM_FACTOR[idx]}")


# In[ ]:


shuffled_indices = np.arange(len(data_train))
np.random.shuffle(shuffled_indices)

input_params_train = np.stack([
    data_train[i].to_numpy(dtype=np.float32) for i in INPUT_PARAMS_ORDER
], axis=-1)
input_params_train /= INPUT_PARAMS_NORM_FACTOR

input_params_train = input_params_train[shuffled_indices]
input_params_train = np.reshape(input_params_train, (-1, BATCH_SIZE_TRAIN, input_params_train.shape[-1]))
print(f"Input params train {input_params_train.shape}")

cell_states_train = np.stack([
    np.array(data_train[cs].to_list())
    for cs in CELL_STATES_ORDER
], axis=-1)
cell_states_train /= CELL_STATES_NORM_FACTOR

cell_states_train = cell_states_train[shuffled_indices]
cell_states_train = np.reshape(
    cell_states_train, (-1, BATCH_SIZE_TRAIN, cell_states_train.shape[-2], cell_states_train.shape[-1])
)
print(f"Cell states train {cell_states_train.shape}")


# In[ ]:


shuffled_indices = np.arange(len(data_test))
np.random.shuffle(shuffled_indices)

input_params_test = np.stack([
    data_test[i].to_numpy(dtype=np.float32) for i in INPUT_PARAMS_ORDER
], axis=-1)
input_params_test /= INPUT_PARAMS_NORM_FACTOR

input_params_test = input_params_test[shuffled_indices]
input_params_test = np.reshape(input_params_test, (-1, BATCH_SIZE_TEST, input_params_test.shape[-1]))
print(f"Input params test {input_params_test.shape}")

cell_states_test = np.stack([
    np.array(data_test[cs].to_list())
    for cs in CELL_STATES_ORDER
], axis=-1)
cell_states_test /= CELL_STATES_NORM_FACTOR

cell_states_test = cell_states_test[shuffled_indices]
cell_states_test = np.reshape(
    cell_states_test, (-1, BATCH_SIZE_TEST, cell_states_test.shape[-2], cell_states_test.shape[-1])
)
print(f"Cell states test {cell_states_test.shape}")


# In[ ]:


N_BATCHES_TRAIN = input_params_train.shape[0]
N_BATCHES_TEST = input_params_test.shape[0]
print(f"Number of batches train {N_BATCHES_TRAIN}")
print(f"Number of batches test {N_BATCHES_TEST}")


# In[ ]:


from utils.datasets import WindowedPredictionDatasetContinuous

dl_train = torch.utils.data.DataLoader(
    WindowedPredictionDatasetContinuous(
        input_params_train, cell_states_train, WINDOW_SIZE
    ), shuffle=False
)
dl_test = torch.utils.data.DataLoader(
    WindowedPredictionDatasetContinuous(
        input_params_test, cell_states_test, 2 # WINDOW_SIZE
    ), shuffle=False
)


# In[ ]:


print(f"Window size: {WINDOW_SIZE}")
print()
print("Training:")
print(f"\t{N_BATCHES_TRAIN} batches with {BATCH_SIZE_TRAIN} sequences of {len(dl_train)} windows")
print("Testing:")
print(f"\t{N_BATCHES_TEST} batches with {BATCH_SIZE_TEST} sequences of {len(dl_test)} windows")


# In[ ]:


for d in dl_train:
    inp, _, cells, labels = d
    for i, c, l in zip(inp.squeeze(dim=0), cells.squeeze(dim=0), labels.squeeze(dim=0)):
        print(i.shape, c.shape, l.shape)
    # print(inp.squeeze().shape, c.squeeze().shape, l.squeeze().shape)
    print()


# In[ ]:


for d in dl_test:
    inp, _, cells, labels = d
    for i, c, l in zip(inp.squeeze(dim=0), cells.squeeze(dim=0), labels.squeeze(dim=0)):
        print(i.shape, c.shape, l.shape)
    # print(inp.squeeze().shape, c.squeeze().shape, l.squeeze().shape)
    print()


# ##### Initialize Model

# In[ ]:


cell_states_dim = len(CELL_STATES_ORDER)
input_params_dim = len(INPUT_PARAMS_ORDER)
n_layers_in = args.n_layers_in
n_layers_out = args.n_layers_in
n_layers_rec = 1
latent_dim = args.latent_dim
dropout_linear = args.dropout_lin
dropout_rec = args.dropout_rec
optimizer_name = 'rmsprop' # 'adam' or 'rmsprop'
nonlinearity = 'relu' if ARCH.lower()=='rnn' else 'tanh'

LEARNING_RATE = 1e-6 if SINGLE_EXP else 1e-6
lr_reduction_factor = 0.1
lr_scheduler_patience = 25
lr_scheduler_threshold = 1e-4
weights_regularization = 0.0

EPOCHS = args.epochs

SAVE_EVERY = 1_500

CLIP_VALUE = 1.


# In[ ]:


MODEL_NAME = f'{ARCH.upper()}_radius{TUMOR_RADIUS}_augment{AUGMENT_DATA}_s{SEED}_ss{SUBSAMPLE_EVERY}_w{WINDOW_SIZE}'+\
    f'_in{n_layers_in}_out{n_layers_out}_lat{latent_dim}'+\
    f'_do{dropout_linear}_dorec{dropout_rec}_nl{nonlinearity}'+\
    f'_wreg{weights_regularization}_opt{optimizer_name}'

print(MODEL_NAME)


# In[ ]:


if ARCH.lower() == 'rnn':
    from architectures.rnn import RNN
    model = RNN(
        cell_states_dim, input_params_dim,
        latent_dim = latent_dim,
        n_layers_in = n_layers_in,
        n_layers_out = n_layers_out,
        n_layers_rnn = n_layers_rec,
        dropout_linear = dropout_linear,
        dropout_rnn = dropout_rec,
        nonlinearity = nonlinearity,
    ).to(device)
elif ARCH.lower() == 'lstm':
    from architectures.lstm import LSTM
    model = LSTM(
        cell_states_dim, input_params_dim,
        latent_dim = latent_dim,
        n_layers_in = n_layers_in,
        n_layers_out = n_layers_out,
        n_layers_lstm = n_layers_rec,
        dropout_linear = dropout_linear,
        dropout_lstm = dropout_rec,
    ).to(device)
else:
    raise ValueError(f"Unknown architecture {ARCH}")


# In[ ]:


EXP_DIR = os.path.join(EXP_DIR, MODEL_NAME)
print(EXP_DIR)
os.makedirs(EXP_DIR, exist_ok=True)


# In[ ]:


for name, p in model.named_parameters():
    print(f"{name}, shape {p.shape}, requires grad {p.requires_grad}")


# In[ ]:


loss_fn = torch.nn.MSELoss(reduce=False)
loss_l1 = torch.nn.L1Loss()

if optimizer_name.lower() == 'rmsprop':
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=LEARNING_RATE, weight_decay=weights_regularization
    )
elif optimizer_name.lower() == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE, weight_decay=weights_regularization
    )
else:
    raise ValueError(f"Unknown optimizer {optimizer_name}")


lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=lr_reduction_factor,
    patience=lr_scheduler_patience,
    threshold=lr_scheduler_threshold
)


# ##### Train Model

# In[ ]:


if WANDB_LOG:
    wandb.init(
        project=f"metamodel_final",
        config={
            "architecture": ARCH.upper(),
            "dataset": "radius",
            "augment_data": AUGMENT_DATA,
            "tumor_radius": TUMOR_RADIUS,
            "subsample_every": SUBSAMPLE_EVERY,
            "single_exp": SINGLE_EXP,
            "loss": loss_fn.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "cell_states_dim": cell_states_dim,
            "input_params_dim": input_params_dim,
            "latent_dim": latent_dim,
            "n_layers_in": n_layers_in,
            "n_layers_out": n_layers_out,
            "n_layers_rec": n_layers_rec,
            "dropout_linear": dropout_linear,
            "dropout_rec": dropout_rec,
            "nonlinearity": nonlinearity,
            "learning_rate": LEARNING_RATE,
            "lr_reduction_factor": lr_reduction_factor,
            "lr_scheduler_patience": lr_scheduler_patience,
            "lr_scheduler_threshold": lr_scheduler_threshold,
            "weights_regularization": weights_regularization,
            "batch_size": BATCH_SIZE_TRAIN,
            "window_size": WINDOW_SIZE,
            "n_batches_train": N_BATCHES_TRAIN,
            "n_batches_test": N_BATCHES_TEST,
            "epochs": EPOCHS,
            "clip_gradient": CLIP_VALUE,
        }
    )


# In[ ]:


loss_dir = os.path.join(EXP_DIR, "loss")
l1_trend_dir = os.path.join(EXP_DIR, "l1_trend")
ncells_trend_dir = os.path.join(EXP_DIR, "ncells_trend")
examples_dir = os.path.join(EXP_DIR, "examples")

os.makedirs(loss_dir, exist_ok=True)
os.makedirs(l1_trend_dir, exist_ok=True)
os.makedirs(ncells_trend_dir, exist_ok=True)
os.makedirs(examples_dir, exist_ok=True)


# In[ ]:


from utils.training import train_epoch_bptt, test_epoch_bptt
from utils.plots import plot_loss, plot_l1loss_trend, plot_ncells_rel_trend
from utils.metrics import cell_states_abserr

avg_loss_list = []
avg_vloss_list = []

avg_epoch_time = 0
avg_val_time = 0

# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ],
#     # record_shapes=False,
#     profile_memory=True,
# ) as prof:
for epoch in range(EPOCHS):
    start = time.time()

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()
    avg_loss = train_epoch_bptt(
        model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        dataloader=dl_train,
        n_batches=N_BATCHES_TRAIN,
        clip_value=CLIP_VALUE,
    )
    
    # We don't need gradients on to do reporting
    start_val = time.time()
    model.eval()
    avg_vloss, input_params_trend, _, cell_states_trend, cell_states_pred =\
        test_epoch_bptt(
            model,
            loss_fn=loss_fn,
            device=device,
            dataloader=dl_test,
            n_batches=N_BATCHES_TEST,
        )
    end_val = time.time()
    lr_scheduler.step(avg_vloss)

    avg_loss_list.append(avg_loss)
    avg_vloss_list.append(avg_vloss)

    abserr_dict = cell_states_abserr(
        cell_states_trend,
        cell_states_pred,
        CELL_STATES_ORDER,
        CELL_STATES_NORM_FACTOR
    )
    
    abserr_dict['loss/l1_start'] = sum([abserr_dict[f"{cs}/abserr_start"] for cs in CELL_STATES_ORDER])
    abserr_dict['loss/l1_mid'] = sum([abserr_dict[f"{cs}/abserr_mid"] for cs in CELL_STATES_ORDER])
    abserr_dict['loss/l1_end'] = sum([abserr_dict[f"{cs}/abserr_end"] for cs in CELL_STATES_ORDER])

    if WANDB_LOG:
        wandb.log({
            **{
                "loss/train": avg_loss,
                "loss/valid": avg_vloss,
            },
            **abserr_dict
        })
    
    if (epoch+1)%SAVE_EVERY==0:
        print("Saving model...")
        torch.save(
            model,
            os.path.join(EXP_DIR, f"{ARCH.lower()}_epoch{epoch+1:03d}.pth")
        )
        plot_loss(
            avg_loss_list,
            avg_vloss_list,
            'MSE Loss',
            loss_dir,
            f'loss_epoch{epoch+1:03d}.png'
        )
        plot_l1loss_trend(
            cell_states_trend,
            cell_states_pred,
            CELL_STATES_ORDER,
            CELL_STATES_NORM_FACTOR,
            TIMESTEP,
            TUMOR_RADIUS,
            l1_trend_dir,
            f'l1loss_trend_epoch{epoch+1:03d}.png'
        )
        plot_ncells_rel_trend(
            cell_states_trend,
            cell_states_pred,
            CELL_STATES_NORM_FACTOR,
            TIMESTEP,
            ncells_trend_dir,
            f'ncells_rel_trend_epoch{epoch+1:03d}.png'
        )

    end = time.time()
    deltat = end-start
    avg_epoch_time += deltat

    deltat_val = end_val - start_val
    avg_val_time += deltat_val

    if (epoch+1)%200 == 0:
        print(f'EPOCH {epoch+1}:')
        print(f'LOSS train {avg_loss} valid {avg_vloss}')
        print(f"{avg_epoch_time/(epoch+1)} avg seconds per epoch")
        print(f"{avg_val_time/(epoch+1)} avg seconds per validation")
        print(flush=True)

if WANDB_LOG:
    wandb.finish()

# profiler_table = prof.key_averages().table(row_limit=-1)
# print(profiler_table)
# print(f"\ntime.time average epoch time: {avg_epoch_time/EPOCHS} s\n")

# with open(os.path.join(EXP_DIR, 'profiler.txt'), 'w') as f:
#     f.write(profiler_table)
#     f.write(f"\n\ntime.time average epoch time: {avg_epoch_time/EPOCHS} s\ntime.time total time: {avg_epoch_time} s\n")

# In[ ]:


# plot sample experiments and model predictions
from utils.plots import plot_sample_experiments

model.eval()
_, input_params_trend, _, cell_states_trend, cell_states_pred = \
    test_epoch_bptt(
        model,
        loss_fn=loss_fn,
        device=device,
        dataloader=dl_test,
        n_batches=N_BATCHES_TEST,
    )

plot_sample_experiments(
    input_params_trend,
    INPUT_PARAMS_ORDER,
    INPUT_PARAMS_NORM_FACTOR,
    cell_states_trend,
    cell_states_pred,
    CELL_STATES_ORDER,
    CELL_STATES_NORM_FACTOR,
    min(5, N_BATCHES_TEST),
    9,
    TUMOR_RADIUS,
    TIMESTEP,
    examples_dir,
    filename=f'exp',
)


# In[ ]:




