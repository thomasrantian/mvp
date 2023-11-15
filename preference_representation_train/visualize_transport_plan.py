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

encoder_path = "/home/thomastian/workspace/visual_representation_alignment_exp_data/franka_push_preference_encoders/6_7_resnet_franka_push_obs_encoder.pt"
obs_encoder = models.Resnet18LinearEncoderNet(**kwargs).cpu()

#obs_encoder.add_activation_layer()
obs_encoder.load_state_dict(torch.load(encoder_path))

# For tcc rep.
checkpoint_dir = "/home/thomastian/workspace/mvp_exp_data/tcc_model/checkpoints/1001.ckpt"
checkpoint = torch.load(checkpoint_dir)
obs_encoder.load_state_dict(checkpoint['model'])


image_size = 112


# # MVP encoder config
# encoder_cfg = {'name': 'vits-mae-hoi', 'pretrain_dir':
#                '/home/thomastian/workspace/mvp_exp_data/mae_encoders', 'freeze': True, 'emb_dim': 128}
# emb_dim = encoder_cfg["emb_dim"]
# obs_encoder = Encoder(
#     model_name=encoder_cfg["name"],
#     pretrain_dir=encoder_cfg["pretrain_dir"],
#     freeze=encoder_cfg["freeze"],
#     emb_dim=emb_dim
# ).cpu()


obs_encoder.eval()

sinkorn_layer = OptimalTransportLayer(gamma=1)



demo_path = '/home/thomastian/workspace/mvp_exp_data/rl_runs/9_12_OT_Kuka_datasize_exp/150/491b5e1f-864c-4b2e-8645-56345664bb85/train_sample/1/'

expert_path = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/6_1_franka_push/contrastive_ranking_triplet/60/positive'



sample_a_embs = get_demo_embs_resnet(demo_path) # 45 x embd_size
sample_b_embs = get_demo_embs_resnet(expert_path) # 45 x embd_size



pe = torch.zeros(45, 32)
position = torch.arange(0, 45, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, 32, 2).float() * (-np.log(10000.0) / 32))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
sample_a_embs = sample_a_embs + pe * 0
sample_b_embs = sample_b_embs + pe * 0

# # Build 45 tensor with 1, 2, 3, ..., 45
# pe = torch.arange(0, 45).unsqueeze(1).float() * 0.00
# # Concatenate the pe with the sample_a_embs and sample_b_embs
# sample_a_embs = torch.cat((sample_a_embs, pe), dim=1)
# sample_b_embs = torch.cat((sample_b_embs, pe), dim=1)




# Get cost matrix for samples using critic network.
cost_matrix = cosine_distance(sample_a_embs, sample_b_embs)
transport_plan = sinkorn_layer(cost_matrix)

# Visualize the transport plan as a heatmap using seaborn
ax = sns.heatmap(transport_plan.detach().cpu().numpy())
plt.show()

ot_rewards = torch.diag(
    torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()
print(ot_rewards)
print(np.sum(ot_rewards))

# Plot the ot rewards
plt.plot(ot_rewards)
plt.show()


# # save the transport plan
# np.save('kuka_tcc_plan_positive_example_1.npy', transport_plan.detach().cpu().numpy())

# save the rewards
np.save('kuka_tcc_reward_move_one_by_one.npy', ot_rewards)