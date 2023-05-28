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

def attention_to_image(attentions, image_size, patch_size, save_dir):
    '''Convert the attention to an image'''
    '''attention: nh x 196'''
    '''return: nh x image_size x image_size'''
    w_featmap = image_size // patch_size
    h_featmap = image_size // patch_size
    nh = attentions.shape[0] # number of head
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()
    plt.imsave(fname=save_dir, arr=attentions[0], format='png')

    return attentions
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
#obs_encoder.load_state_dict(torch.load('/home/thomastian/workspace/mvp_exp_data/mae_encoders/frankapush_obs_encoder.pt'))
# Set the obs_encoder to train mode
obs_encoder.eval()

test_image_path = '/home/thomastian/workspace/temp/42.png'
test_image = Image.open(test_image_path)
test_image_tensor = pixel_to_tensor(np.asarray(test_image))[0].cpu().float()

# Get the original observation encoder attention
obs_encoder_ave_attention_mvp = get_image_encoder_ave_attention(obs_encoder, test_image_tensor)
# Save the image
output_dir = '/home/thomastian/workspace/mvp_exp_data/encoder_ave_att_image.png'
encoder_ave_att_image = attention_to_image(obs_encoder_ave_attention_mvp, 224, 16, output_dir)


obs_encoder.load_state_dict(torch.load('/home/thomastian/workspace/mvp_exp_data/mae_encoders/frankapush_obs_encoder.pt'))
# Get the original observation encoder attention
obs_encoder_ave_attention_preference = get_image_encoder_ave_attention(obs_encoder, test_image_tensor)
# Save the image
output_dir = '/home/thomastian/workspace/mvp_exp_data/encoder_ave_att_image_ours.png'
encoder_ave_att_image = attention_to_image(obs_encoder_ave_attention_preference, 224, 16, output_dir)

print(obs_encoder_ave_attention_mvp - obs_encoder_ave_attention_preference)