import os
import numpy as np
import cv2
from PIL import Image
import torch
import sys
sys.path.append("..")
from ddn.ddn.pytorch.optimal_transport import sinkhorn, OptimalTransportLayer
import torch.nn as nn
import models
from resnet_representation_alignment_util_old import *

# Set the sequence length of the demonstration. For now: 50 for sweep, 30 for collision avoidance
sequence_length = 45

# Extract the train data from the directory
contrastive_ranking_data_dir = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/6_1_franka_push/contrastive_ranking_triplet'
contrastive_ranking_data = extract_data_from_dir('contrastive', contrastive_ranking_data_dir, sequence_length, 'resnet').cuda()
equal_ranking_data_dir = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/6_1_franka_push/equal_ranking_triplet'
equal_ranking_data = extract_data_from_dir('equal_ranking', equal_ranking_data_dir, sequence_length, 'resnet').cuda()

# Make sure contrastive ranking data and equal ranking data are the same size
# contrastive_ranking_data = contrastive_ranking_data[0:equal_ranking_data.shape[0], :, :, :, :, :]



# Split contrastive ranking data into train and eval. The slip is 80:20
contrastive_ranking_data_size = contrastive_ranking_data.shape[0]
train_size = int(contrastive_ranking_data_size * 0)
eval_size = contrastive_ranking_data_size - train_size
train_contrastive_ranking_data = contrastive_ranking_data[0:train_size, :, :, :, :, :]
eval_contrastive_ranking_data = contrastive_ranking_data[train_size:, :, :, :, :, :]


enable_equal_ranking = True
enable_batch_processing = True
enable_feature_aligner = False

# Initialize the obs_encoder (Resnet18)
kwargs = {
    "num_ctx_frames": 1,
    "normalize_embeddings": False,
    "learnable_temp": False,
    "embedding_size": 32,
}
obs_encoder = models.Resnet18LinearEncoderNet(**kwargs).cuda()
# Set the obs_encoder to train mode
# Load the Resnet encoder and load the pre-trained tcc weight (if empty, then load the default weight)
encoder_path = "/home/thomastian/workspace/mvp_exp_data/preference_encoders/8_30_resnet_franka_push_obs_encoder_datasize50.pt"
obs_encoder.load_state_dict(torch.load(encoder_path))
obs_encoder.eval()


# Initialize the feature_aligner
device = torch.device('cuda:0')
feature_aligner = FeatureAligner(sequence_length=sequence_length, embed_dim=32, n_heads=1, n_layers=1, device=device).cuda()

# The batch size used for the training
set_batch_size = 8
sinkorn_layer = OptimalTransportLayer(gamma = 1)

if enable_feature_aligner:
    optimizer = torch.optim.Adam(list(feature_aligner.parameters()) + list(obs_encoder.parameters()), lr=1e-4)
else:
    optimizer = torch.optim.Adam(obs_encoder.parameters(), lr=1e-3)

# To do: move them to util function
def get_alignment_score(eval_contrastive_ranking_data, eval_equal_ranking_data=None):
    '''Evaluate the model on a batch of data'''
    with torch.no_grad():
        running_loss = 0.0
        num_triplets_evaled = 0
        correct_preds = 0
        for i in range(0, eval_contrastive_ranking_data.shape[0], set_batch_size):
            # Extract batch from contrastive train data
            current_batch = eval_contrastive_ranking_data[i:i+set_batch_size] # B x 3 x T x 3 x 112 x 112
            batch_ot_reward_positive_neutral, batch_ot_reward_positive_negative = get_batch_ot_reward(current_batch, sequence_length, enable_batch_processing)
            # Compute the loss for the contrastive ranking (negative probability of positive)
            batch_loss_contrastive = 1-torch.exp(batch_ot_reward_positive_neutral) / (torch.exp(batch_ot_reward_positive_neutral) + torch.exp(batch_ot_reward_positive_negative)) # B x 1

            # Count the number of the examples that have values smaller than 0.4
            correct_preds += torch.sum(batch_loss_contrastive < 0.4).item()
            num_triplets_evaled += batch_loss_contrastive.shape[0]
    return correct_preds / num_triplets_evaled

def get_batch_ot_reward(current_batch, sequence_length, batch_process):
    """Compute the OT reward for a batch of data"""
    '''Input shape: B x 3 x T x 3 x 112 x 112'''
    batch_size = current_batch.shape[0]    
    # If batch_process is True, use obs_encoder to directly process the batch. If False, use the encoder to process each element in the batch
    if batch_process:
        # Convert the batch to 4 dim
        current_batch = current_batch.view(batch_size * 3, sequence_length, 3, 112, 112) # (B x 3) x T x 3 x 112 x 112
        # Use obs_encoder to encode the batch
        current_batch = obs_encoder(current_batch) # (B x 3) x T x 32
        if enable_feature_aligner:
            # Use the Feature Aligner to do cross-attention on the batch
            current_batch = feature_aligner(current_batch.embs)
        else:
            current_batch = current_batch.embs
    else:
        current_batch = current_batch.view(batch_size * 3 * sequence_length, 3, 112, 112) # (B x 3 x T) x 3 x 112 x 112
        current_batch_obs_embd = []
        for i in range(current_batch.shape[0]):
            current_frame = current_batch[i].unsqueeze(0).unsqueeze(0)  # 1 x 1 x 3 x 112 x 112
            current_frame_obs_embd = obs_encoder(current_frame).embs # 1 x 1 x 32
            current_batch_obs_embd.append(current_frame_obs_embd)
        # Stack the list of embeddings to a tensor
        current_batch = torch.stack(current_batch_obs_embd, dim=0) # (B x 3 x T) x 1 x 32
        # Reshape the batch to (B x 3) x T x 32
        current_batch = current_batch.view(batch_size * 3, sequence_length, 32) # (B x 3) x T x 32
        if enable_feature_aligner:
            # Use the Feature Aligner to do cross-attention on the batch
            current_batch = feature_aligner(current_batch) # (B x 3) x T x 32
    
    # Reshape the batch to B x 3 x T x 32
    current_batch = current_batch.view(batch_size, 3, sequence_length, 32) # B x 3 x T x embedding_dim
    # Compute the cost-matrix of the batch: (positive, neutral) and (positive, negative), both have the same shape B x T x T
    current_batch_positive = current_batch[:, 0, :, :] # B x T x 32
    current_batch_negative = current_batch[:, 1, :, :] # B x T x 32
    current_batch_neutral = current_batch[:, 2, :, :] # B x T x 32
    batch_cost_matrix_positive_neutral =  batch_cosine_distance(current_batch_positive, current_batch_neutral) 
    batch_cost_matrix_positive_negative = batch_cosine_distance(current_batch_positive, current_batch_negative)
    # Compute the transport plan of the batch
    batch_transport_plan_postive_neutral = sinkorn_layer(batch_cost_matrix_positive_neutral)
    batch_transport_plan_postive_negative = sinkorn_layer(batch_cost_matrix_positive_negative)
    
    # Compute batch_ot_reward_positive_neutral. Use torcch unbind to unbind the batch into a list of B tensors
    temp = torch.unbind(torch.bmm(batch_transport_plan_postive_neutral, batch_cost_matrix_positive_neutral.transpose(1, 2)), dim=0)
    batch_ot_transport_cost = torch.block_diag(*temp) # (BxT) x (BxT)
    batch_ot_reward_positive_neutral = torch.sum(-torch.diag(batch_ot_transport_cost).view(batch_size, sequence_length), dim=1) # B x 1
    
    # Compute batch_ot_reward_positive_negative
    temp = torch.unbind(torch.bmm(batch_transport_plan_postive_negative, batch_cost_matrix_positive_negative.transpose(1, 2)), dim=0)
    batch_ot_transport_cost = torch.block_diag(*temp) # (BxT) x (BxT)
    batch_ot_reward_positive_negative = torch.sum(-torch.diag(batch_ot_transport_cost).view(batch_size, sequence_length), dim=1) # B x 1

    return batch_ot_reward_positive_neutral, batch_ot_reward_positive_negative


alignment_score = get_alignment_score(eval_contrastive_ranking_data)
print('Alignment score: ', alignment_score)