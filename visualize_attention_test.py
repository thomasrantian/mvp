#!/usr/bin/env python3

import numpy as np
import os
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal
import torchvision
import vit2 as vit

from PIL import Image
import matplotlib.pyplot as plt

_MODELS = {
    "vits-mae-hoi": "mae_pretrain_hoi_vit_small.pth",
    "vits-mae-in": "mae_pretrain_imagenet_vit_small.pth",
    "vits-sup-in": "sup_pretrain_imagenet_vit_small.pth",
    "vitb-mae-egosoup": "mae_pretrain_egosoup_vit_base.pth",
    "vitl-256-mae-egosoup": "mae_pretrain_egosoup_vit_large_256.pth",
}
_MODEL_FUNCS = {
    "vits": vit.vit_s16,
    "vitb": vit.vit_b16,
    "vitl": vit.vit_l16,
}


device = 'cpu'

class Encoder(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze, emb_dim):
        super(Encoder, self).__init__()
        assert model_name in _MODELS, f"Unknown model name {model_name}"
        model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        img_size = 256 if "-256-" in model_name else 224
        pretrain_path = os.path.join(pretrain_dir, _MODELS[model_name])
        self.backbone, gap_dim = model_func(pretrain_path, img_size=img_size)
        if freeze:
            self.backbone.freeze()
        self.freeze = freeze
        self.projector = nn.Linear(gap_dim, emb_dim)

    @torch.no_grad()
    def forward(self, x):
        feat = self.backbone.extract_feat(x)
        return self.projector(self.backbone.forward_norm(feat)), feat

    def forward_feat(self, feat):
        return self.projector(self.backbone.forward_norm(feat))

encoder_cfg = {'name': 'vits-mae-hoi', 'pretrain_dir': \
    '/home/thomastian/workspace/mvp', 'freeze': True, 'emb_dim': 32}

emb_dim = encoder_cfg["emb_dim"]

obs_enc = Encoder(
            model_name=encoder_cfg["name"],
            pretrain_dir=encoder_cfg["pretrain_dir"],
            freeze=encoder_cfg["freeze"],
            emb_dim=emb_dim
        )
# print(obs_enc)
# obs_emb, obs_feat = obs_enc(torch.randn(1, 3, 224, 224))
# print(obs_emb.shape)

def pixel_to_tensor(arr):
    arr = torch.from_numpy(arr).permute(2, 0, 1).float()[None, None, Ellipsis]
    arr = arr / 255.0
    arr = arr.to(device)
    return arr

frame_path = '/home/thomastian/workspace/mvp_old/mvp_exp_data/attention_test/148.png'
image_observation = Image.open(frame_path)
image_observation.show()
image_observation = np.asarray(image_observation)
image_tensor = pixel_to_tensor(image_observation)[0]
obs_emb, obs_feat = obs_enc(image_tensor)
print(obs_feat.shape)

attentions = obs_enc.backbone.get_last_selfattention(image_tensor)

print(attentions.shape)
nh = attentions.shape[1] # number of head

# we keep only the output patch attention
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

w_featmap = 224 // 16
h_featmap = 224 // 16

attentions = attentions.reshape(nh, w_featmap, h_featmap)
attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()

output_dir = '/home/thomastian/workspace/mvp_old/mvp_exp_data/attention_test'

# os.makedirs(args.output_dir, exist_ok=True)
# torchvision.utils.save_image(torchvision.utils.make_grid(image_observation, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
for j in range(nh):
    fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
    plt.imsave(fname=fname, arr=attentions[j], format='png')
    print(f"{fname} saved.")

# def get_frame_embs(frame_path):
#     """Get embeddings for a single frame."""
#     """Return a 1 x emb_dim tensor."""
#     image_observation = Image.open(frame_path)
#     #image_observation.show()
#     image_observation = np.asarray(image_observation)
#     image_tensor = pixel_to_tensor(image_observation)[0]
#     obs_emb, obs_feat = obs_enc(image_tensor)
#     return obs_feat

# def get_demo_embs(demo_path):
#     """Get embeddings for all frames in a demo.""" 
#     """Return a n_frames x emb_dim tensor."""
#     total_frames_in_demo = len(os.listdir(demo_path))
#     current_demo_embs = []
#     for frame_id in range(total_frames_in_demo):
#         frame_path = os.path.join(demo_path, str(frame_id+1)) + '.png'
#         image_embs = get_frame_embs(frame_path).detach().numpy()
#         current_demo_embs.append(image_embs)
#     current_demo_embs = np.concatenate(current_demo_embs, axis=0)
#     return current_demo_embs


# negative_demo_embs = get_demo_embs('/home/thomastian/workspace/mvp/mvp_exp_data/hand_view_negative')
# positive_demo_embs = get_demo_embs('/home/thomastian/workspace/mvp/mvp_exp_data/hand_view_positive')


# std_feature_id = []
# for i in range(positive_demo_embs.shape[1]):
#     std = np.std(positive_demo_embs[:,i])
#     std_feature_id.append([i, std])
# std_feature_id = np.array(std_feature_id)
# # sort by std
# sorted_indices = np.argsort(std_feature_id[:,1])
# # flip the order to descending
# #sorted_indices = np.flip(sorted_indices, axis=0)
# std_feature_id = std_feature_id[sorted_indices]

# # Create a new figure and axis
# # Plot each dimension as a curve
# for i in range(10):
#     plt.plot(positive_demo_embs[:,sorted_indices[i]],'o',label='pos-dim {}'.format(sorted_indices[i]))
#     plt.plot(negative_demo_embs[:,sorted_indices[i]],'-',label='neg-dim {}'.format(sorted_indices[i]))


# plt.legend()
# #plt.ylim(-20, 0)
# plt.xlabel('Time step')
# plt.ylabel('Embdedding value')
# plt.show()
