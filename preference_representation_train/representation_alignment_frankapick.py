import os
import numpy as np
import cv2
from PIL import Image
import torch
from ddn.ddn.pytorch.optimal_transport import sinkhorn, OptimalTransportLayer
import torch.nn as nn

from representation_alignment_util import *

# Set the sequence length of the demonstration. For now: 50 for sweep, 30 for collision avoidance
sequence_length = 35

# Extract the train data from the directory
contrastive_ranking_data_dir = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/5_17_franka_pick/contrastive_ranking_triplet'
contrastive_ranking_data = extract_data_from_dir('contrastive', contrastive_ranking_data_dir, sequence_length).cuda()
equal_ranking_data_dir = '/home/thomastian/workspace/mvp_exp_data/representation_model_train_data/5_17_franka_pick/equal_ranking_triplet'
equal_ranking_data = extract_data_from_dir('equal_ranking', equal_ranking_data_dir, sequence_length).cuda()

# Make sure contrastive ranking data and equal ranking data are the same size
# contrastive_ranking_data = contrastive_ranking_data[0:equal_ranking_data.shape[0], :, :, :, :, :]

# Split contrastive ranking data into train and eval. The slip is 80:20
contrastive_ranking_data_size = contrastive_ranking_data.shape[0]
train_size = int(contrastive_ranking_data_size * 0.7)
eval_size = contrastive_ranking_data_size - train_size
train_contrastive_ranking_data = contrastive_ranking_data[0:train_size, :, :, :, :, :]
eval_contrastive_ranking_data = contrastive_ranking_data[train_size:, :, :, :, :, :]
# Split equal ranking data into train and eval. The slip is 80:20
equal_ranking_data_size = equal_ranking_data.shape[0]
train_size = int(equal_ranking_data_size * 0.7)
eval_size = equal_ranking_data_size - train_size
train_equal_ranking_data = equal_ranking_data[0:train_size, :, :, :, :, :]
eval_equal_ranking_data = equal_ranking_data[train_size:, :, :, :, :, :]

enable_equal_ranking = True
enable_batch_processing = True
enable_feature_aligner = False

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
).cuda()
# Set the obs_encoder to train mode
obs_encoder.train()

def _freeze_module(m):
    for name, p in m.named_parameters():
        # if name != '0.weight':
        p.requires_grad = False

# Only make the last layer trainable
#_freeze_module(obs_encoder.backbone)
trainable_params = []
for name, p in obs_encoder.named_parameters():
    if p.requires_grad:
        trainable_params.append(name)

print("Trainable parameters in the encoder:")
print(trainable_params)

# Initialize the feature_aligner
device = torch.device('cuda:0')
#feature_aligner = FeatureAligner(sequence_length=sequence_length, embed_dim=32, n_heads=1, n_layers=1, device=device).cuda()

# The batch size used for the training
set_batch_size = 2
sinkorn_layer = OptimalTransportLayer(gamma = 1)

if enable_feature_aligner:
    optimizer = torch.optim.Adam(list(feature_aligner.parameters()) + list(obs_encoder.parameters()), lr=1e-4)
else:
    optimizer = torch.optim.Adam(obs_encoder.parameters(), lr=1e-2)

# To do: move them to util function
def eval_batch(eval_contrastive_ranking_data, eval_equal_ranking_data):
    '''Evaluate the model on a batch of data'''
    obs_encoder.eval()
    with torch.no_grad():
        running_loss = 0.0
        num_triplets_evaled = 0
        for i in range(0, eval_contrastive_ranking_data.shape[0], set_batch_size):
            # Extract batch from contrastive train data
            current_batch = eval_contrastive_ranking_data[i:i+set_batch_size] # B x 3 x T x 3 x 112 x 112
            batch_ot_reward_positive_neutral, batch_ot_reward_positive_negative = get_batch_ot_reward(current_batch, sequence_length, enable_batch_processing)
            # Compute the loss for the contrastive ranking (negative probability of positive)
            batch_loss_contrastive = torch.norm(1-torch.exp(batch_ot_reward_positive_neutral) / (torch.exp(batch_ot_reward_positive_neutral) + torch.exp(batch_ot_reward_positive_negative))) # B x 1

            if enable_equal_ranking:
                # Extract batch from equal train data
                current_batch = eval_equal_ranking_data[i:i+set_batch_size] # B x 3 x T x 3 x 112 x 112
                batch_ot_reward_positive_neutral, batch_ot_reward_positive_negative = get_batch_ot_reward(current_batch, sequence_length, enable_batch_processing)
                # Compute the loss for the equal ranking
                batch_loss_equal =  torch.norm(0.5 - torch.exp(batch_ot_reward_positive_neutral) / (torch.exp(batch_ot_reward_positive_neutral) + torch.exp(batch_ot_reward_positive_negative))) # B x 1
            else:
                batch_loss_equal = batch_loss_contrastive 
            
            # Compute the sum of the batch loss
            loss = torch.sum(batch_loss_contrastive + batch_loss_equal)
            running_loss += loss.item()
            num_triplets_evaled = num_triplets_evaled + current_batch.shape[0]
    obs_encoder.train()
    return running_loss / num_triplets_evaled / 2

def get_batch_ot_reward(current_batch, sequence_length, batch_process):
    """Compute the OT reward for a batch of data"""
    '''Input shape: B x 3 x T x 3 x 112 x 112'''
    batch_size = current_batch.shape[0]    
    # If batch_process is True, use obs_encoder to directly process the batch. If False, use the encoder to process each element in the batch
    if batch_process:
        # Convert the batch to 4 dim
        current_batch = current_batch.view(batch_size * 3 * sequence_length, 3, h, w) # (B x 3 x T) x 3 x 112 x 112
        # Use obs_encoder to encode the batch
        current_batch, _ = obs_encoder(current_batch) # (B x 3 x T) x 128
        if enable_feature_aligner:
            # Use the Feature Aligner to do cross-attention on the batch
            current_batch = feature_aligner(current_batch.embs)
        else:
            current_batch = current_batch
    else:
        current_batch = current_batch.view(batch_size * 3 * sequence_length, 3, h, w) # (B x 3 x T) x 3 x 112 x 112
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
    current_batch = current_batch.view(batch_size, 3, sequence_length, 128) # B x 3 x T x embedding_dim
    # Compute the cost-matrix of the batch: (positive, neutral) and (positive, negative), both have the same shape B x T x T
    current_batch_positive = current_batch[:, 0, :, :] # B x T x 128
    current_batch_negative = current_batch[:, 1, :, :] # B x T x 128
    current_batch_neutral = current_batch[:, 2, :, :] # B x T x 128
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


best_eval_loss = 10000

for epoch in range(50):
    running_loss = 0.0
    num_triplets_evaled = 0
    
    for i in range(0, train_contrastive_ranking_data.shape[0], set_batch_size):
        # Extract batch from contrastive train data
        current_batch = train_contrastive_ranking_data[i:i+set_batch_size] # B x 3 x T x 3 x 224 x 224
        batch_ot_reward_positive_neutral, batch_ot_reward_positive_negative = get_batch_ot_reward(current_batch, sequence_length, enable_batch_processing)
        # Compute the loss for the contrastive ranking (negative probability of positive)
        batch_loss_contrastive =  torch.norm(1-torch.exp(batch_ot_reward_positive_neutral) / (torch.exp(batch_ot_reward_positive_neutral) + torch.exp(batch_ot_reward_positive_negative))) # B x 1
        if enable_equal_ranking:
            # Extract batch from contrastive train data
            current_batch = train_equal_ranking_data[i:i+set_batch_size] # B x 3 x T x 3 x 112 x 112
            batch_ot_reward_positive_neutral, batch_ot_reward_positive_negative = get_batch_ot_reward(current_batch, sequence_length, enable_batch_processing)
            # Compute the loss for the equal ranking
            batch_loss_equal =  torch.norm(0.5 - torch.exp(batch_ot_reward_positive_neutral) / (torch.exp(batch_ot_reward_positive_neutral) + torch.exp(batch_ot_reward_positive_negative))) # B x 1
        else:
            batch_loss_equal = batch_loss_contrastive 
        
        # Compute the sum of the batch loss
        loss = torch.sum(batch_loss_contrastive + batch_loss_equal)
        
        if epoch > 1:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        num_triplets_evaled = num_triplets_evaled + current_batch.shape[0]
    print('Epoch: %d, average per-triplet loss: %.3f' % (epoch + 1, running_loss / num_triplets_evaled / 2))
    
    # Running validation
    obs_encoder.eval()
    if enable_feature_aligner:
        feature_aligner.eval()
    val_loss = eval_batch(eval_contrastive_ranking_data, eval_equal_ranking_data)
    obs_encoder.train()
    if enable_feature_aligner:
        feature_aligner.train()
    print('Validation loss: %.3f' % val_loss)
    if val_loss < best_eval_loss:
        best_eval_loss = val_loss
        if enable_feature_aligner:
            torch.save(feature_aligner.state_dict(), '430_feature_aligner_weight_with_aligner.pt')
        torch.save(obs_encoder.state_dict(), 'frankapick_obs_encoder.pt')
        print('Model saved.')