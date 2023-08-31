import os
import numpy as np
import cv2
from PIL import Image
import torch
from ddn.ddn.pytorch.optimal_transport import sinkhorn, OptimalTransportLayer
import torch.nn as nn
import models
from resnet_representation_alignment_util import *


def extract_one_rollout(data_dir):
    '''Extract one rollout from the data_dir
    output: T x 3 x 224 x 224'''
    # Find the number of frames in the dir
    total_frames_in_demo = len(os.listdir(data_dir))
    all_frames_in_demo = []
    # Extract the frame
    for frame_id in range(45):
        frame_path = os.path.join(data_dir, str(frame_id)) + '.png'
        image_observation = Image.open(frame_path)
        image_observation = np.asarray(image_observation)
        image_observation = cv2.resize(image_observation, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
        image_tensor = pixel_to_tensor(image_observation, resnest_im_mean, resnest_im_std).squeeze() # 3 x 112 x 112
        # If the image is not 112 x 112, resize it
        if image_tensor.shape[1] != 112 or image_tensor.shape[2] != 112:
            image_tensor = torch.nn.functional.interpolate(image_tensor, size=(112, 112), mode='bilinear', align_corners=False)
        all_frames_in_demo.append(image_tensor)
    # Stack all the frames in the demo
    all_frames_in_demo = torch.stack(all_frames_in_demo, dim=0) # T x 3 x 112 x 112
    #all_frames_in_demo = all_frames_in_demo[::3, :, :, :] # (T/3) x 112 x 112
    return all_frames_in_demo

def get_demo_embs(demo_path):
    roll_out = extract_one_rollout(demo_path).unsqueeze(0) # 1 x T x 3 x 112 x 112
    current_demo_embs = obs_encoder.infer(roll_out).embs.unsqueeze(0) # T x 128
    #current_demo_embs = feature_aligner(current_demo_embs)
    return current_demo_embs.squeeze()


# Initialize the obs_encoder (Resnet18)
kwargs = {
      "num_ctx_frames": 1,
      "normalize_embeddings": False,
      "learnable_temp": False,
      "embedding_size": 32,
  }
obs_encoder = models.Resnet18LinearEncoderNet(**kwargs).cpu()
# Set the obs_encoder to train mode
# Load the Resnet encoder and load the pre-trained tcc weight (if empty, then load the default weight)
pretrained_path = "/home/thomastian/workspace/mvp_exp_data/tcc_model/checkpoints/1001.ckpt"

checkpoint_dir = "/home/thomastian/workspace/mvp_exp_data/tcc_model_human/checkpoints/201.ckpt"
checkpoint = torch.load(checkpoint_dir)
#obs_encoder.load_state_dict(checkpoint['model'])
obs_encoder.load_state_dict(torch.load('/home/thomastian/workspace/mvp/8_12_resnet_human_push_obs_encoder.pt'))
obs_encoder.eval()


for i in range(0, 40):
    sample_a = '/home/thomastian/workspace/mvp_exp_data/behavior_train_data/8_12_human_push/0'
    sample_b = '/home/thomastian/workspace/mvp_exp_data/rl_runs/8_12_OT_kuka/8eaae63e-6ffe-4cef-bf0d-e64e14e94c45/good_sample/' + str(i)
    sample_a = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/6_1_franka_push/contrastive_ranking_triplet/1/positive'
    #sample_b = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/6_1_franka_push/contrastive_ranking_triplet/1/negative'


    sample_a_embs = get_demo_embs(sample_a)
    sample_b_embs = get_demo_embs(sample_b)

    sinkorn_layer = OptimalTransportLayer(gamma = 1)
    cost_matrix = cosine_distance(sample_b_embs, sample_a_embs)  # Get cost matrix for samples using critic network.
    transport_plan = sinkorn_layer(cost_matrix)
    ot_rewards = torch.diag(torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()
    print(i, np.sum(ot_rewards))
