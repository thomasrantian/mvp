
import os
import numpy as np
from PIL import Image
import torch
import shutil


meta_demo_folder = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/6_1_franka_push/meta_demo/'

target_dir = '/home/thomastian/workspace/mvp_exp_data/mae_train_robot_data/'

# if the target directory does not exist, create it.
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

n_images_saved = 0

for i in range(100):
    for image in range(45):
        image_path = os.path.join(meta_demo_folder, str(i), str(image) + '.png')
        image_save_name = str(n_images_saved) + '.png'
        image_save_path = os.path.join(target_dir, image_save_name)
        shutil.copy(image_path, image_save_path)
        n_images_saved += 1