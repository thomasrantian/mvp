import os
import numpy as np
import cv2
import shutil

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


# Define the path to the processed positive demos folder
processed_positive_demo_path = '/home/thomastian/workspace/mvp_exp_data/human_demo/positive_demo/'
# Define the path to the processed negative demos folder
processed_negative_demo_path = '/home/thomastian/workspace/mvp_exp_data/human_demo/negative_demo/'
# Define the path to the processed neutral demos folder
processed_neutral_demo_path = '/home/thomastian/workspace/mvp_exp_data/human_demo/neutral_demo/'

# Define the train data save path
train_data_save_path = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/8_12_human_push/contrastive_ranking_triplet'
# Check if the train data save path exists, if not, create it
if not os.path.exists(train_data_save_path):
    os.makedirs(train_data_save_path)

# Count the total number of demos
total_positive_demos = len(os.listdir(processed_positive_demo_path))
total_neutral_demos = len(os.listdir(processed_neutral_demo_path))
total_negative_demos = len(os.listdir(processed_negative_demo_path))

triple_id = 0

# Loop through all the positive demos, neutral demos and negative demos
for positive_demo_id in range(total_positive_demos):
    for neutral_demo_id in range(total_neutral_demos):
        for negative_demo_id in range(total_negative_demos):
            positive_demo_path = os.path.join(processed_positive_demo_path, str(positive_demo_id))
            neutral_demo_path = os.path.join(processed_neutral_demo_path, str(neutral_demo_id))
            negative_demo_path = os.path.join(processed_negative_demo_path, str(negative_demo_id))
            # Define the path to the current triple
            current_triple_path = os.path.join(train_data_save_path, str(triple_id))
            # Define the path to the positive folder in the triple
            current_positive_path = os.path.join(current_triple_path, 'positive')
            # Define the path to the neutral folder in the triple
            current_neutral_path = os.path.join(current_triple_path, 'neutral')
            # Define the path to the negative folder in the triple
            current_negative_path = os.path.join(current_triple_path, 'negative')
            # Copy the positive demo to the triplet positive folder
            copy_file_to_dir(positive_demo_path, current_positive_path)
            # Copy the neutral demo to the triplet neutral folder
            copy_file_to_dir(neutral_demo_path, current_neutral_path)
            # Copy the negative demo to the triplet negative folder
            copy_file_to_dir(negative_demo_path, current_negative_path)
            # Increment the triple id
            triple_id += 1

