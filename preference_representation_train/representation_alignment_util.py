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

#from extraction import *

# define some constants used in the code
w = 224
h = 224

im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device='cpu').view(3, 1, 1)
im_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device='cpu').view(3, 1, 1)

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

# MVP Encoder


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

    #@torch.no_grad()
    def forward(self, x):
        feat = self.backbone.extract_feat(x)
        return self.projector(self.backbone.forward_norm(feat)), feat

    def forward_feat(self, feat):
        return self.projector(self.backbone.forward_norm(feat))

# FeatureAligner is a transformer encoder that takes in patch embeddings from the MVP encoder,
# append a classification token to the beginning of the sequence, and then use the transformer encoder to align the features.
# The output from the classificaiton token is the final representation used to predict the possitive and negative class


class FeatureAligner(nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers, dropout=0.1):
        super(FeatureAligner, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        # Add a linear layer to project the output of the transformer encoder to the number of classes
        self.classifier = nn.Linear(emb_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.cls_token = nn.Parameter(torch.ones(1, 1, self.emb_dim))

    def forward(self, x):
        # Use the image encoder to extract the patch embeddings
        #x  = self.obs_enc.backbone.extract_encoded_patch_embed(x)
        # Append a classification token to the beginning of the sequence
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.transformer(x)
        # use the output from the classification token to predict the class
        x = self.classifier(x[:, 0, :])
        return torch.sigmoid(x)
    

    def get_feature_aligner_attention(self, x):
        '''Returns the attention weights from the last transformer encoder layer.'''
        '''output size: B  x 196'''
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.transformer(x)
        attn_output_weights = self.transformer.layers[-1].attn_output_weights # B x 197 x 197
        # we keep only the CLS  patch attention
        attn_output_weights = attn_output_weights[0, 0, 1:].reshape(1, -1)  #1 x 196
        return attn_output_weights



class MAPFeatureAligner(nn.Module):
    def __init__(self, n_latents, emb_dim):
        super(MAPFeatureAligner, self).__init__()
        self.emb_dim = emb_dim
        self.n_latents = n_latents
        self.map_block = MAPBlock(self.n_latents, self.emb_dim, self.n_latents)
        # Add a linear layer to project the output of the transformer encoder to the number of classes
        self.classifier = nn.Linear(emb_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Run the MAP block
        x = self.map_block(x) # B x emb_dim
        x = self.classifier(x)
        return torch.sigmoid(x)
    

    def get_feature_aligner_attention(self, x):
        '''Returns the attention weights from the last transformer encoder layer.'''
        '''output size: B  x 196'''
        x = self.map_block(x) # B x n_latents x emb_dim
        cross_attention = self.map_block.attn.cross_attention # B x 2(head) x 196
        # average the attention weights from the two heads
        cross_attention = torch.mean(cross_attention, dim=1) # B x 196
        return cross_attention

# To do: add the decoder feature aligner

def get_image_encoder_ave_attention(obs_enc, x):
    '''Returns the average attention weights from the last image encoder layer.'''
    '''output size: B  x 196'''
    encoder_attentions  = obs_enc.backbone.get_last_selfattention(x) #B x n_head x 197 x 197
    nh = encoder_attentions.shape[1]  # number of head
    # we keep only the CLS  patch attention
    encoder_attentions = encoder_attentions[0, :, 0, 1:].reshape(nh, -1) #n_head x 196
    # Return the average attention weights
    return torch.mean(encoder_attentions, dim=0).unsqueeze(0)

def pixel_to_tensor(arr):
    '''Converts a image numpy array to a torch tensor.'''
    arr = torch.from_numpy(arr).permute(2, 0, 1)
    arr = arr / 255.0
    # subtract the mean and divide by the standard deviation
    arr = (arr - im_mean) / im_std
    return arr.float()[None, None, Ellipsis]


def prepare_train_data():
    all_demo_frames_data = []
    all_demo_frames_label = []

    # Iterate through all the folders in the data set directory
    for demo_class in ['positive', 'negative']:
        demo_data_set_dir = '/home/thomastian/workspace/mvp/mvp_fine_tune_data/cabinet/' + demo_class
        total_available_demos = len(os.listdir(demo_data_set_dir))
        for i in range(total_available_demos):
            # Find the path to the folder
            demo_path = os.path.join(demo_data_set_dir, str(i))
            # Find the number of images in the folder
            total_frames = len(os.listdir(demo_path))
            # Iterate through all the images in the folder
            for j in range(total_frames):
                # if j < 100:
                #     continue
                # Find the path to the image
                frame_path = os.path.join(demo_path, str(j+1) + '.png')
                # Open the image
                frame = Image.open(frame_path)
                # Convert the image to a tensor
                frame = pixel_to_tensor(np.asarray(frame))[0]
                # Append the tensor to the list
                all_demo_frames_data.append(frame)
                # Append the label to the list
                if demo_class == 'positive':
                    all_demo_frames_label.append([1])
                else:
                    all_demo_frames_label.append([0])

    # Shuffle the data and label lists together
    zipped = list(zip(all_demo_frames_data, all_demo_frames_label))
    random.shuffle(zipped)
    all_demo_frames_data, all_demo_frames_label = zip(*zipped)

    # Split the data and label lists into training and testing sets
    train_demo_frames_data = all_demo_frames_data[:int(
        len(all_demo_frames_data)*0.7)]
    train_demo_frames_label = all_demo_frames_label[:int(
        len(all_demo_frames_label)*0.7)]
    test_demo_frames_data = all_demo_frames_data[int(
        len(all_demo_frames_data)*0.7):]
    test_demo_frames_label = all_demo_frames_label[int(
        len(all_demo_frames_label)*0.7):]

    # Convert the data and label lists into tensors
    train_demo_frames_data = torch.stack(
        train_demo_frames_data, dim=0).squeeze().cuda().float()
    train_demo_frames_label = torch.tensor(
        train_demo_frames_label).cuda().float()
    test_demo_frames_data = torch.stack(
        test_demo_frames_data, dim=0).squeeze().cuda().float()
    test_demo_frames_label = torch.tensor(
        test_demo_frames_label).cuda().float()
    
    # Return the training and testing data and labels
    return train_demo_frames_data, train_demo_frames_label, test_demo_frames_data, test_demo_frames_label


def compute_accuracy(pred, label):
    '''Computes the accuracy of the predictions.'''
    pred = pred > 0.5
    return (pred == label).sum().float() / len(pred)


def eval_batch(obs_enc, feature_aligner, data_input, data_labels, batch_size, alignment_mode):
    '''Evaluate the model on a batch of data'''
    running_accuracy = 0.0
    num_samples = data_input.shape[0]
    with torch.no_grad():
        num_batch = 0
        for i in range(0, num_samples, batch_size):
            # Prepare batch data
            inputs = data_input[i:i+batch_size]
            labels = data_labels[i:i+batch_size]
            if alignment_mode == 'fine_tune':
                outputs = obs_enc.backbone.forward_classifier(inputs)
            if alignment_mode == 'augment':
                # Encode the image [Image is already normalized during the data generation]
                encoded_patch_embed = obs_enc.backbone.extract_encoded_patch_embed(
                    inputs)
                # Forward pass
                outputs = feature_aligner(encoded_patch_embed)

            running_accuracy += compute_accuracy(outputs, labels)
            num_batch += 1
    return running_accuracy / num_batch




def extract_frames_from_dir(data_dir, n_frames):
    '''Extract one rollout images from the data_dir
    output: T x 3 x 112 x 112'''
    all_frames_in_demo = []
    # Extract the frame
    for frame_id in range(n_frames):
        frame_path = os.path.join(data_dir, str(frame_id)) + '.png'
        image_observation = Image.open(frame_path)
        image_observation = np.asarray(image_observation)
        image_observation = cv2.resize(image_observation, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        image_tensor = pixel_to_tensor(image_observation).squeeze() # 3 x 112 x 112 and normalized the pixel value
        all_frames_in_demo.append(image_tensor)
    # Stack all the frames in the demo
    all_frames_in_demo = torch.stack(all_frames_in_demo, dim=0) # T x 3 x 112 x 112
    #all_frames_in_demo = all_frames_in_demo[::3, :, :, :] # (T/3) x 112 x 112
    return all_frames_in_demo

def get_batch_data(sequence_length, contrastive_ranking_data_dir, contrastive_indexs, equal_ranking_data_dir, equal_indexs):
    """Return the batch data given the indexs"""
    contrastive_ranking_data = extract_data_from_dir('contrastive', contrastive_ranking_data_dir, sequence_length, contrastive_indexs).cuda()
    equal_ranking_data = extract_data_from_dir('equal_ranking', equal_ranking_data_dir, sequence_length, equal_indexs).cuda()
    return contrastive_ranking_data, equal_ranking_data

def extract_data_from_dir(data_mode, data_dir, sequence_length, indexs):
    '''Extract the data from the data_dir'''
    '''Data mode can be equal or contrast'''
    '''Output:  N_data x 3 x T x 3 x 112 x 112'''
    # Find the total number of tripletd in the data_dir
    total_num_triplets = len(os.listdir(data_dir))
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
        triplet_rollouts = torch.stack(triplet_rollouts, dim=0) # 3 x T x 3 x 112 x 112
        # Append the triplet rollouts to the data
        data.append(triplet_rollouts)
     
    data = torch.stack(data, dim=0) # N_data x 3 x T x 3 x 112 x 112
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