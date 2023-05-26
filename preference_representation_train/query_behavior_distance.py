import os
import numpy as np
import cv2
from PIL import Image
import torch
from ddn.ddn.pytorch.optimal_transport import sinkhorn, OptimalTransportLayer
import torch.nn as nn

from representation_alignment_util import *


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
        image_observation = cv2.resize(image_observation, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        image_tensor = pixel_to_tensor(image_observation).squeeze() # 3 x 112 x 112
        all_frames_in_demo.append(image_tensor)
    # Stack all the frames in the demo
    all_frames_in_demo = torch.stack(all_frames_in_demo, dim=0) # T x 3 x 112 x 112
    #all_frames_in_demo = all_frames_in_demo[::3, :, :, :] # (T/3) x 112 x 112
    return all_frames_in_demo

def get_demo_embs(demo_path):
    roll_out = extract_one_rollout(demo_path) # T x 3 x 112 x 112
    current_demo_embs,_ = obs_encoder(roll_out) # T x 128
    #current_demo_embs = feature_aligner(current_demo_embs)
    return current_demo_embs


# Initialize the obs_encoder (Resnet18)
# MVP encoder config
encoder_cfg = {'name': 'vits-mae-hoi', 'pretrain_dir':
               '/home/thomastian/workspace/mvp_exp_data/mae_encoders', 'freeze': True, 'emb_dim': 128}
emb_dim = encoder_cfg["emb_dim"]
obs_encoder = Encoder(
    model_name=encoder_cfg["name"],
    pretrain_dir=encoder_cfg["pretrain_dir"],
    freeze=encoder_cfg["freeze"],
    emb_dim=emb_dim
).cpu()
obs_encoder.load_state_dict(torch.load('/home/thomastian/workspace/mvp/frankapush_obs_encoder.pt'))
# Set the obs_encoder to train mode
obs_encoder.eval()

# sample_a = '/home/thomastian/workspace/mvp_exp_data/behavior_train_data/franka_pick/3'
# sample_b = '/home/thomastian/workspace/mvp_exp_data/rl_runs/d86248f0-f1c8-42a4-ab2b-28b9bf1f103a/train_sample/22'

sample_a = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/5_24_franka_pick_push/contrastive_ranking_triplet/1/positive'
sample_b = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/5_24_franka_pick_push/contrastive_ranking_triplet/1/negative'


sample_a_embs = get_demo_embs(sample_a)
sample_b_embs = get_demo_embs(sample_b)

sinkorn_layer = OptimalTransportLayer(gamma = 1)
cost_matrix = cosine_distance(sample_b_embs, sample_a_embs)  # Get cost matrix for samples using critic network.
transport_plan = sinkorn_layer(cost_matrix)
ot_rewards = torch.diag(torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()
print(np.sum(ot_rewards))
