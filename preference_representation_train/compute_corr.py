import scipy.stats as stats
import numpy as np

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from PIL import Image
import torch
from ddn.ddn.pytorch.optimal_transport import sinkhorn, OptimalTransportLayer
import torch.nn as nn
import models
from resnet_representation_alignment_util_old import *
import seaborn as sns


def extract_one_rollout_resnet(data_dir):
    '''Extract one rollout from the data_dir
    output: T x 3 x 112 x 112'''
    # Find the number of frames in the dir
    total_frames_in_demo = len(os.listdir(data_dir))
    all_frames_in_demo = []
    # Uniformly sample 15 frames from the demo
    for frame_id in range(0, 45, 1):
        frame_path = os.path.join(data_dir, str(frame_id)) + '.png'
        image_observation = Image.open(frame_path)
        image_observation = np.asarray(image_observation)
        image_observation = cv2.resize(image_observation, dsize=(
            image_size, image_size), interpolation=cv2.INTER_CUBIC)
        image_tensor = pixel_to_tensor(
            image_observation, resnest_im_mean, resnest_im_std).squeeze()  # 3 x 112 x 112
        # If the image is not 112 x 112, resize it
        if image_tensor.shape[1] != image_size or image_tensor.shape[2] != image_size:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor, size=(image_size, image_size), mode='bilinear', align_corners=False)
        all_frames_in_demo.append(image_tensor)
    # Stack all the frames in the demo
    all_frames_in_demo = torch.stack(
        all_frames_in_demo, dim=0)  # T x 3 x 112 x 112
    return all_frames_in_demo

def extract_one_rollout_vit(data_dir):
    '''Extract one rollout images from the data_dir
    output: T x 3 x 224 x 224'''
    total_frames_in_demo = len(os.listdir(data_dir))
    all_frames_in_demo = []
    # Extract the frame
    for frame_id in range(0, 45, 1):
        frame_path = os.path.join(data_dir, str(frame_id)) + '.png'
        image_observation = Image.open(frame_path)
        image_observation = np.asarray(image_observation)
        image_observation = cv2.resize(image_observation, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        image_tensor = pixel_to_tensor(image_observation, vit_im_mean, vit_im_std).squeeze() # 3 x 224 x 224
        all_frames_in_demo.append(image_tensor)
    # Stack all the frames in the demo
    all_frames_in_demo = torch.stack(all_frames_in_demo, dim=0) # T x 3 x 224 x 224
    return all_frames_in_demo

def get_demo_embs_resnet(demo_path):
    roll_out = extract_one_rollout_resnet(
        demo_path).unsqueeze(0)  # 1 x T x 3 x 112 x 112
    current_demo_embs = obs_encoder.infer(
        roll_out).embs.unsqueeze(0)  # T x 128
    #current_demo_embs = feature_aligner(current_demo_embs)
    return current_demo_embs.squeeze()

def get_demo_embs_vit(demo_path):
    roll_out = extract_one_rollout_vit(demo_path) # T x 3 x 224 x 224
    current_demo_embs, _ = obs_encoder(roll_out)
    return current_demo_embs.squeeze()


# Initialize the obs_encoder (Resnet18)
kwargs = {
    "num_ctx_frames": 1,
    "normalize_embeddings": False,
    "learnable_temp": False,
    "embedding_size": 32,
}
encoder_path = "/home/thomastian/workspace/visual_representation_alignment_exp_data/franka_push_preference_encoders/8_30_resnet_franka_push_obs_encoder_datasize150.pt"
obs_encoder = models.Resnet18LinearEncoderNet(**kwargs).cpu()

#obs_encoder.add_activation_layer()
#obs_encoder.load_state_dict(torch.load(encoder_path))

# # For tcc rep.
# checkpoint_dir = "/home/thomastian/workspace/mvp_exp_data/tcc_model/checkpoints/1001.ckpt"
# checkpoint = torch.load(checkpoint_dir)
# obs_encoder.load_state_dict(checkpoint['model'])
obs_encoder.eval()

sinkorn_layer = OptimalTransportLayer(gamma=1)


image_size = 112


P_set = []

expert_path = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/6_1_franka_push/contrastive_ranking_triplet/60/positive/'



def find_best_ot_reward(demo_path):
    
    dist_2_exp_list = []

    reward_list = []

    for i in range(5):
        expert_path = '/home/thomastian/workspace/mvp_exp_data/behavior_train_data/6_1_franka_push/' + str(i) + '/'
        sample_a_embs = get_demo_embs_resnet(demo_path) # 45 x embd_size
        sample_b_embs = get_demo_embs_resnet(expert_path) # 45 x embd_size

        # Get cost matrix for samples using critic network.
        cost_matrix = cosine_distance(sample_a_embs, sample_b_embs)
        transport_plan = sinkorn_layer(cost_matrix)

        ot_rewards = torch.diag(
            torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()

        dist_2_exp = np.sum(ot_rewards)

        dist_2_exp_list.append(dist_2_exp)
        reward_list.append(ot_rewards)
    
    # return the reward that has the smallest distance to the expert
    return reward_list[np.argmin(dist_2_exp_list)]



# Given a numpy array A and a numpy array B, compute the spearman correlation between the A and B using scipy.stats.spearmanr
def compute_spearmanr(A, B):
    return stats.spearmanr(A, B)[0]

for n in range(0, 50):


    demo_path = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/6_1_franka_push/contrastive_ranking_triplet/' + str(n) + '/negative/'

    ot_rewards = find_best_ot_reward(demo_path)

    # sample_a_embs = get_demo_embs_resnet(demo_path) # 45 x embd_size
    # sample_b_embs = get_demo_embs_resnet(expert_path) # 45 x embd_size

    # # Get cost matrix for samples using critic network.
    # cost_matrix = cosine_distance(sample_a_embs, sample_b_embs)
    # transport_plan = sinkorn_layer(cost_matrix)

    # ot_rewards = torch.diag(
    #     torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()

    true_pref_reward = np.load('/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/6_1_franka_push/contrastive_ranking_triplet/' + str(n) + '/negative/true_pref_reward_hist.npy')

    # Down sample the true_pref_reward and ot_rewards
    true_pref_reward = true_pref_reward[::4]
    ot_rewards = ot_rewards[::4]

    pho = compute_spearmanr(true_pref_reward, ot_rewards)
    P_set.append(pho)
    print(pho)

print(np.mean(P_set))