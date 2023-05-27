
import os
import numpy as np
from PIL import Image
import shutil
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# exlucde the last folder
DIR_PATH = os.path.dirname(DIR_PATH)
DIR_PATH = os.path.dirname(DIR_PATH)
# Copy every file in the source directory to target directory.
def copy_file_to_dir(source_dir, target_dir):
    # If the target directory does not exist, create it. Else, delete it and create a new one.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)

    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        if os.path.isfile(file_path):
            shutil.copy(file_path, target_dir)

# Define the path to the meta data directory.
data_dir = DIR_PATH + '/mvp_exp_data/representation_model_train_data/5_26_franka_push'
meta_data_dir = os.path.join(data_dir, 'meta_demo')
# Find the total file number in the directory.
total_available_demo = len(os.listdir(meta_data_dir))
data_gen_mode = 'contrastive_ranking_triplet'
#data_gen_mode = 'equal_ranking_triplet'


# Go through every demo in the meta data directory and save the reward to a list.
reward_list = []
for demo_id in range(total_available_demo):
    reward_path = os.path.join(meta_data_dir, str(demo_id), 'sum_true_dense_reward.npy')
    reward_list.append(np.load(reward_path))
reward_list = np.array(reward_list)
# Plot the reward distribution.
import matplotlib.pyplot as plt
plt.hist(reward_list, bins=30)
plt.show()


# data source directory
if data_gen_mode == 'equal_ranking_triplet':
    equal_ranking_data_save_dir = os.path.join(data_dir, 'equal_ranking_triplet')
    # Check if the dir exists.If not, create it.
    if not os.path.exists(equal_ranking_data_save_dir):
        os.makedirs(equal_ranking_data_save_dir)

    num_triplet_found = 0
    start_folder_name = 100
    while num_triplet_found < 100:
        # Random sample three rollouts from the data directory.
        demo_list = np.random.choice(total_available_demo, 3, replace=False)
        # Extract the reward.npy file from each rollout.
        reward_list = []
        last_step_reward_list = []
        for demo_id in demo_list:
            reward_path = os.path.join(meta_data_dir, str(demo_id), 'sum_true_dense_reward.npy')
            reward_list.append(np.load(reward_path))
            reward_hist_path = os.path.join(meta_data_dir, str(demo_id), 'true_dense_reward_hist.npy')
            last_step_reward_list.append(np.load(reward_hist_path)[-1])
        # Convert the reward list to numpy array.
        reward_list = np.array(reward_list)
        last_step_reward_list = np.array(last_step_reward_list)
        # Sort the reward list and the demo list based on the reward. The first one is the lowest reward.
        reward_list, demo_list, last_step_reward_list = zip(*sorted(zip(reward_list, demo_list, last_step_reward_list))) 
        #if reward_list[0] < -40 and reward_list[1] < -40 and reward_list[2] < -40:
        if reward_list[0] > -35 and reward_list[1] > -35 and  reward_list[2] > -35:
            # Create a directory to store the three rollouts. The directory name id the number of triplet found.
            triplet_save_dir = os.path.join(equal_ranking_data_save_dir, str(start_folder_name))
            # Check if the dir exists.If not, create it. If exists, delete it and create a new one.
            if not os.path.exists(triplet_save_dir):
                os.makedirs(triplet_save_dir)
            else:
                shutil.rmtree(triplet_save_dir)
                os.makedirs(triplet_save_dir)
            # Copy the three rollouts to the triplet directory.
            for i in range(3):
                demo_id = demo_list[i]
                # Creat a directory to store the rollout.
                rollout_save_dir = os.path.join(triplet_save_dir, str(i))
                # Check if the dir exists.If not, create it.
                if not os.path.exists(rollout_save_dir):
                    os.makedirs(rollout_save_dir)
                # Copy the rollout to the directory.
                copy_file_to_dir(os.path.join(meta_data_dir, str(demo_id)), rollout_save_dir)
            # Update the number of triplet found.
            num_triplet_found += 1
            start_folder_name += 1

if data_gen_mode == 'contrastive_ranking_triplet':
    contrastive_ranking_data_save_dir = os.path.join(data_dir, 'contrastive_ranking_triplet')
    # Check if the dir exists.If not, create it.
    if not os.path.exists(contrastive_ranking_data_save_dir):
        os.makedirs(contrastive_ranking_data_save_dir)
    num_triplet_found = 0
    start_folder_name = 0
    reward_diff_threshold = 8
    while num_triplet_found < 200:
        # Random sample three rollouts from the data directory.
        demo_list = np.random.choice(total_available_demo, 3, replace=False)
        # Extract the reward.npy file from each rollout.
        reward_list = []
        last_step_reward_list = []
        for demo_id in demo_list:
            reward_path = os.path.join(meta_data_dir, str(demo_id), 'sum_true_dense_reward.npy')
            reward_list.append(np.load(reward_path))
            reward_hist_path = os.path.join(meta_data_dir, str(demo_id), 'true_dense_reward_hist.npy')
            last_step_reward_list.append(np.load(reward_hist_path)[-2])
        # Convert the reward list to numpy array.
        reward_list = np.array(reward_list)
        last_step_reward_list = np.array(last_step_reward_list)
        # Check if the three rewards are different at least by 2.
        def _reward_diff(r1, r2, threshold):
            return np.abs(r1 - r2) > threshold 
        #if _reward_diff(reward_list[0], reward_list[1], reward_diff_threshold) and _reward_diff(reward_list[0], reward_list[2], reward_diff_threshold) and _reward_diff(reward_list[1], reward_list[2], reward_diff_threshold):
        # Sort the reward list and the demo list based on the reward. The first one is the lowest reward.
        reward_list, demo_list, last_step_reward_list = zip(*sorted(zip(reward_list, demo_list, last_step_reward_list)))
        #if reward_list[0] <= -30 and (reward_list[1] >= -20 and reward_list[1] < 0) and reward_list[2] >= -15:
        #if _reward_diff(reward_list[0], reward_list[1], reward_diff_threshold) and _reward_diff(reward_list[0], reward_list[2], reward_diff_threshold) and _reward_diff(reward_list[1], reward_list[2], reward_diff_threshold):    
        '''sweep only condition 50 demos for each condition'''
        #if (reward_list[0] <= -30 and abs(last_step_reward_list[0]) <= 0.7 and abs(last_step_reward_list[0]) >=0.45) and (reward_list[1] >= -25 and reward_list[1] < -10 and abs(last_step_reward_list[1]) < 0.05) and reward_list[2] >= -10:
        if (reward_list[0] <= -40) and (reward_list[1] >= -35 and reward_list[1] < -30) and reward_list[2] >= -30:
            # Create a directory to store the three rollouts. The directory name id the number of triplet found.
            triplet_save_dir = os.path.join(contrastive_ranking_data_save_dir, str(start_folder_name))
            # Check if the dir exists.If not, create it. If exists, delete it and create a new one.
            if not os.path.exists(triplet_save_dir):
                os.makedirs(triplet_save_dir)
            else:
                shutil.rmtree(triplet_save_dir)
                os.makedirs(triplet_save_dir)
            # Copy the three rollouts to the triplet directory.
            roll_out_name = ['negative', 'neutral', 'positive']
            for i in range(3):
                demo_id = demo_list[i]
                # Creat a directory to store the rollout.
                rollout_save_dir = os.path.join(triplet_save_dir, roll_out_name[i])
                # Check if the dir exists.If not, create it.
                if not os.path.exists(rollout_save_dir):
                    os.makedirs(rollout_save_dir)
                # Copy the rollout to the directory.
                copy_file_to_dir(os.path.join(meta_data_dir, str(demo_id)), rollout_save_dir)
            # Update the number of triplet found.
            num_triplet_found += 1
            start_folder_name += 1

