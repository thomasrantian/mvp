
import os
import numpy as np
from PIL import Image
import torch
import shutil

# Copy every file in the source directory to target directory.
def copy_file_to_dir(source_dir, target_dir):
    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        if os.path.isfile(file_path):
            shutil.copy(file_path, target_dir)

# Source directory.
source_dir = '/home/thomastian/workspace/mvp_exp_data/rl_runs/1efb720e-1cb2-4954-a3d0-f2c51341a275/train_sample'
# Target directory.
target_dir = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/5_17_franka_pick/meta_demo'


# We want to copy every file in the source directory to target directory. And rename the file name starting from n_file.
# Find the total number of files in the target directory.
n_file = len(os.listdir(target_dir))
i = 0
# Go through every folder in the source directory, and copy it to the target directory.
for folder in os.listdir(source_dir):
    if i % 1 == 0:
        folder_path = os.path.join(source_dir, folder)
        # Load the dist_to_expert_min
        #dist_to_expert_min = np.load(os.path.join(folder_path, 'dist_to_expert_min.npy'))
        # Load the reward
        sum_reward = np.load(os.path.join(folder_path, 'sum_true_dense_reward.npy'))
        # Load the reward hist
        reward_hist = np.load(os.path.join(folder_path, 'true_dense_reward_hist.npy'))
        if True:
        #if sum_reward < -30:
        #if abs(reward_hist[-1]) < 0.1:
            new_folder_path = os.path.join(target_dir, str(n_file))
            os.makedirs(new_folder_path)
            # Copy every file in the folder to the new folder.
            copy_file_to_dir(folder_path, new_folder_path)
            # Increment the n_file.
            n_file += 1
    i += 1

