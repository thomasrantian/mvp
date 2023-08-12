
import os
import numpy as np
from PIL import Image
import torch
import shutil
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# exlucde the last folder
DIR_PATH = os.path.dirname(DIR_PATH)
DIR_PATH = os.path.dirname(DIR_PATH)

# Copy every file in the source directory to target directory.
def copy_file_to_dir(source_dir, target_dir):
    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        if os.path.isfile(file_path):
            # if the file is a png file, copy it to the target directory.
            if file_path.endswith('.png'):
                shutil.copy(file_path, target_dir)

# Source directory.
source_dir = DIR_PATH + '/mvp_exp_data/representation_model_train_data/6_1_franka_push/meta_demo'
# Target directory.
target_dir = DIR_PATH + '/mvp_exp_data/tcc_train/train/mediumstick'

# if the target directory does not exist, create it.
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# We want to copy every file in the source directory to target directory. And rename the file name starting from n_file.
# Find the total number of files in the target directory.
n_file = len(os.listdir(target_dir))
i = 0
# Go through every folder in the source directory, and copy it to the target directory.
for folder in os.listdir(source_dir):
    if i % 1 == 0:# and i > 1000:
        folder_path = os.path.join(source_dir, folder)
        # Load the dist_to_expert_min
        #dist_to_expert_min = np.load(os.path.join(folder_path, 'sum_ot_reward.npy'))
        # Load the reward
        #sum_reward = np.load(os.path.join(folder_path, 'sum_true_dense_reward.npy'))
        # Load the reward hist
        #reward_hist = np.load(os.path.join(folder_path, 'true_dense_reward_hist.npy'))
        preference_reward_path = np.load(os.path.join(folder_path, 'true_pref_reward_hist.npy'))
        last_reward = preference_reward_path[-2]
        #if True:
        if last_reward < 0.1:
        #if abs(dist_to_expert_min) < 0.5 and last_reward > 0.1:
            new_folder_path = os.path.join(target_dir, str(n_file))
            os.makedirs(new_folder_path)
            # Copy every file in the folder to the new folder.
            copy_file_to_dir(folder_path, new_folder_path)
            # Increment the n_file.
            n_file += 1
    i += 1

