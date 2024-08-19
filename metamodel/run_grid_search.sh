#!/bin/bash

# Define lists of parameters for each argument
# tumor_radius_list=(100 275 50 400)
# subsample_every_list=(1 2 3)
# latent_dim_list=(500 1000)
# dropout_rec_list=(0.2 0.4)
# window_size_list=(5 8 12 24)
n_layers_in_list=(2 3)

# Nested loops to iterate over each parameter list
# for subsample_every in "${subsample_every_list[@]}"; do
#     for latent_dim in "${latent_dim_list[@]}"; do
#         for dropout_rec in "${dropout_rec_list[@]}"; do
#             for window_size in "${window_size_list[@]}"; do
#                 for n_layers_in in "${n_layers_in_list[@]}"; do
#                     python rec_model_train.py --tumor_radius 100 --latent_dim 500 --window_size 12 --n_layers_in $n_layers_in --dropout_lin 0 --dropout_rec 0.2 --subsample_every 3
#                 done
#             done
#         done
#     done
# done

# 2, 1k, 0.2, 24, 2
python rec_model_train.py --tumor_radius 50 --latent_dim 1000 --window_size 24 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2 --subsample_every 2 --augment_data 1
python rec_model_train.py --tumor_radius 100 --latent_dim 1000 --window_size 24 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2 --subsample_every 2 --augment_data 1
python rec_model_train.py --tumor_radius 275 --latent_dim 1000 --window_size 24 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2 --subsample_every 2 --augment_data 1
python rec_model_train.py --tumor_radius 400 --latent_dim 1000 --window_size 24 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2 --subsample_every 2 --augment_data 1

# 1, 1k, 0.2, 48, 2
# python rec_model_train.py --tumor_radius 50 --latent_dim 1000 --window_size 48 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2 --subsample_every 1 --augment_data 1
# python rec_model_train.py --tumor_radius 100 --latent_dim 1000 --window_size 48 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2 --subsample_every 1 --augment_data 1
# python rec_model_train.py --tumor_radius 275 --latent_dim 1000 --window_size 48 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2 --subsample_every 1 --augment_data 1
# python rec_model_train.py --tumor_radius 400 --latent_dim 1000 --window_size 48 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2 --subsample_every 1 --augment_data 1

# 3, 1k, 0.2, 12, 2
# python rec_model_train.py --tumor_radius 100 --latent_dim 1000 --window_size 16 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2 --subsample_every 3 --augment_data 1

# 3, 500, 0.2, 12, 2
# python rec_model_train.py --tumor_radius 100 --latent_dim 500 --window_size 16 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2 --subsample_every 3 --epochs 6000 --augment_data 1
