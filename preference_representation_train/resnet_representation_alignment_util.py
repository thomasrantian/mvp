import numpy as np
import os
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal
import torchvision
import vit2 as vit

from PIL import Image
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import cv2

# define some constants used in the code
resnet_w = 112
resnet_h = 112

vit_w = 224
vit_h = 224

vit_im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device='cpu').view(3, 1, 1)
vit_im_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device='cpu').view(3, 1, 1)

resnest_im_mean = torch.tensor([0, 0, 0], dtype=torch.float, device='cpu').view(3, 1, 1)
resnest_im_std = torch.tensor([1, 1, 1], dtype=torch.float, device='cpu').view(3, 1, 1)


def pixel_to_tensor(arr, im_mean, im_std):
    '''Converts a image numpy array to a torch tensor.'''
    arr = torch.from_numpy(arr).permute(2, 0, 1)
    arr = arr / 255.0
    # subtract the mean and divide by the standard deviation
    arr = (arr - im_mean) / im_std
    return arr.float()[None, None, Ellipsis]

# MVP encoder config
_MODELS = {
    "vits-mae-hoi": "mae_pretrain_hoi_vit_small.pth",
    "vits-mae-hoi-ft": "mae_pretrain_hoi_vit_small_fine_tune.pth",
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

# MVP Encoder class
class Encoder(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze, emb_dim):
        super(Encoder, self).__init__()
        assert model_name in _MODELS, f"Unknown model name {model_name}"
        model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        img_size = 256 if "-256-" in model_name else 224
        pretrain_path = os.path.join(pretrain_dir, _MODELS[model_name])
        self.backbone, gap_dim = model_func(
            pretrain_path, img_size=img_size, num_classes=1)
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

class FeatureAligner(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        device = torch.device('cpu'),
    ) -> None:
        super().__init__()
        self.sequence_length, self.embed_dim, self.n_heads, self.n_layers = sequence_length, embed_dim, n_heads, n_layers
        self.device = device
        # Initialize learnable tokens
        self.learnable_tokens = nn.Parameter(torch.zeros(1, self.sequence_length, self.embed_dim))
        nn.init.normal_(self.learnable_tokens, std=0.02)

        # Add a transformer decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.n_heads, batch_first=True, dropout=0.4)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)

        # Positional encoding
        pe = torch.zeros(self.sequence_length, self.embed_dim, device=self.device)
        position = torch.arange(0, self.sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-np.log(10000.0) / self.embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Repeat the learnable tokens
        learnable_tokens = self.learnable_tokens.repeat(x.shape[0], 1, 1) + self.pe.repeat(x.shape[0], 1, 1)
        # # Repeat the positional encoding
        # pe = self.pe.repeat(x.shape[0], 1, 1)
        # learnable_tokens = learnable_tokens + pe
        x = self.transformer_decoder(learnable_tokens, x)
        return x


def extract_frames_from_dir_resnet(data_dir, n_frames):
    '''Extract one rollout images from the data_dir
    output: T x 3 x 112 x 112'''
    all_frames_in_demo = []
    # Extract the frame
    for frame_id in range(n_frames):
        frame_path = os.path.join(data_dir, str(frame_id)) + '.png'
        image_observation = Image.open(frame_path)
        image_observation = np.asarray(image_observation)
        image_observation = cv2.resize(image_observation, dsize=(resnet_w, resnet_h), interpolation=cv2.INTER_CUBIC)
        image_tensor = pixel_to_tensor(image_observation, resnest_im_mean, resnest_im_std).squeeze() # 3 x 112 x 112
        # If the image is not 112 x 112, resize it
        if image_tensor.shape[1] != 112 or image_tensor.shape[2] != 112:
            image_tensor = torch.nn.functional.interpolate(image_tensor, size=(112, 112), mode='bilinear', align_corners=False)
        all_frames_in_demo.append(image_tensor)
    # Stack all the frames in the demo
    all_frames_in_demo = torch.stack(all_frames_in_demo, dim=0) # T x 3 x 112 x 112
    #all_frames_in_demo = all_frames_in_demo[::3, :, :, :] # (T/3) x 112 x 112
    return all_frames_in_demo

def extract_frames_from_dir_vit(data_dir, n_frames):
    '''Extract one rollout images from the data_dir
    output: T x 3 x 224 x 224'''
    all_frames_in_demo = []
    # Extract the frame
    for frame_id in range(n_frames):
        frame_path = os.path.join(data_dir, str(frame_id)) + '.png'
        image_observation = Image.open(frame_path)
        image_observation = np.asarray(image_observation)
        image_observation = cv2.resize(image_observation, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        image_tensor = pixel_to_tensor(image_observation, vit_im_mean, vit_im_std).squeeze() # 3 x 112 x 112 and normalized the pixel value
        all_frames_in_demo.append(image_tensor)
    # Stack all the frames in the demo
    all_frames_in_demo = torch.stack(all_frames_in_demo, dim=0) # T x 3 x 112 x 112
    #all_frames_in_demo = all_frames_in_demo[::3, :, :, :] # (T/3) x 112 x 112
    return all_frames_in_demo

def get_batch_data(data_mode, encoder_type, sequence_length, contrastive_ranking_data_dir, contrastive_indexs):
    """Return the batch data given the indexs"""
    contrastive_ranking_data = extract_data_from_dir(data_mode, contrastive_ranking_data_dir, sequence_length, encoder_type, contrastive_indexs).cuda()
    return contrastive_ranking_data


def extract_data_from_dir(data_mode, data_dir, sequence_length, encoder_type, indexs):
    '''Extract the data from the data_dir'''
    '''Data mode can be equal or contrast'''
    '''Output:  N_data x 3 x T x 3 x img_s x img_s'''
    
    if encoder_type == 'resnet':
        extract_frames_from_dir = extract_frames_from_dir_resnet
    elif encoder_type == 'vit':
        extract_frames_from_dir = extract_frames_from_dir_vit    
    data = []
    for i in indexs:
        triplet_dir = data_dir + '/' + str(i) + '/'
        if data_mode == 'contrastive':
            positive_rollout_dir = triplet_dir + 'positive/'
            negative_rollout_dir = triplet_dir + 'negative/'
            neutral_rollout_dir = triplet_dir + 'neutral/'
        if data_mode == 'equal_ranking':
            positive_rollout_dir = triplet_dir + '0/'
            negative_rollout_dir = triplet_dir + '1/'
            neutral_rollout_dir = triplet_dir + '2/'
        # Extract the positive rollout
        positive_rollout = extract_frames_from_dir(positive_rollout_dir, sequence_length)
        # Extract the negative rollout
        negative_rollout = extract_frames_from_dir(negative_rollout_dir, sequence_length)
        # Extract the neutral rollout
        neutral_rollout = extract_frames_from_dir(neutral_rollout_dir, sequence_length)
        # Stack the positive, negative and neutral rollouts
        triplet_rollouts = [positive_rollout, negative_rollout, neutral_rollout]
        triplet_rollouts = torch.stack(triplet_rollouts, dim=0) # 3 x T x 3 x img_s x img_s
        # Append the triplet rollouts to the data
        data.append(triplet_rollouts)

    data = torch.stack(data, dim=0) # N_data x 3 x T x 3 x img_s x img_s
    return data

def batch_cosine_distance(x, y):
    C = torch.bmm(x, y.transpose(1, 2))
    x_norm = torch.norm(x, p=2, dim=2)
    y_norm = torch.norm(y, p=2, dim=2)
    x_n = x_norm.unsqueeze(2)
    y_n = y_norm.unsqueeze(2)
    norms = torch.bmm(x_n, y_n.transpose(1, 2))
    C = (1 - C / norms)
    return C

def cosine_distance(x, y):
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    x_n = x_norm.unsqueeze(1)
    y_n = y_norm.unsqueeze(1)
    norms = torch.mm(x_n, y_n.T)
    C = (1 - C / norms)
    return C

def euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, 2))
    return c

# Comupte the euclidean distance between two tensor A = (B x T x D) and B = (B x T x D)
def batch_euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(2)
    y_lin = y.unsqueeze(1)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, 3))
    return c