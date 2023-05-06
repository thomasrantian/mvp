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

# Load from util
from representation_alignment_util import *


# MVP encoder config
encoder_cfg = {'name': 'vits-mae-hoi', 'pretrain_dir':
               '/home/thomastian/workspace/mvp', 'freeze': True, 'emb_dim': 128}
emb_dim = encoder_cfg["emb_dim"]
obs_enc = Encoder(
    model_name=encoder_cfg["name"],
    pretrain_dir=encoder_cfg["pretrain_dir"],
    freeze=encoder_cfg["freeze"],
    emb_dim=emb_dim
).cuda()


# Load the training data
train_demo_frames_data, train_demo_frames_label, test_demo_frames_data, test_demo_frames_label = prepare_train_data()


# Define the aligner type
aligner = 'MAP'

# Initialize the model and move it to the GPU
emb_dim = 384
num_heads = 1
num_layers = 1
dropout = 0.2

if aligner == 'TFEncoder':
    feature_aligner = FeatureAligner(
        emb_dim=384, num_heads=num_heads, num_layers=num_layers, dropout=dropout).cuda()
if aligner == 'MAP':
    feature_aligner = MAPFeatureAligner(n_latents=1, emb_dim=emb_dim).cuda()


run_training_job = False
# #aligment_mode = 'fine_tune'
aligment_mode = 'augment'


 # Train the model
num_epochs = 40
batch_size = 100
num_samples = train_demo_frames_data.shape[0]


if run_training_job:

    # Define optimizer and loss function
    if aligment_mode == 'fine_tune':
        optimizer = torch.optim.Adam(obs_enc.parameters(), lr=0.01)
    if aligment_mode == 'augment':
        optimizer = torch.optim.Adam(feature_aligner.parameters(), lr=0.01)
    criterion = nn.BCELoss().requires_grad_(True)

    # Set the model to training mode
    if aligment_mode == 'fine_tune':
        obs_enc.train()
    if aligment_mode == 'augment':
        obs_enc.eval()
        feature_aligner.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        for i in range(0, num_samples, batch_size):
            # Prepare batch data
            inputs = train_demo_frames_data[i:i+batch_size]
            labels = train_demo_frames_label[i:i+batch_size]
            if aligment_mode == 'fine_tune':
                outputs = obs_enc.backbone.forward_classifier(inputs)
            if aligment_mode == 'augment':
                encoded_patch_embed = obs_enc.backbone.extract_encoded_patch_embed(
                inputs)
                outputs = feature_aligner(encoded_patch_embed)

            loss = criterion(outputs, labels.view(outputs.shape[0], 1))
            #loss = Variable(loss, requires_grad = True)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += compute_accuracy(outputs, labels)

        # Print the current epoch loss and accuracy
        print(
            f"Epoch {epoch+1} loss: {running_loss / (num_samples/batch_size):.4f}")
        print(
            f"Epoch {epoch+1} accuracy: {running_accuracy / (num_samples/batch_size):.4f}")

        # Eval the model every 10 epochs
        if (epoch+1) % 10 == 0:
            # Set the model to evaluation mode
            if aligment_mode == 'fine_tune':
                obs_enc.eval()
            if aligment_mode == 'augment':
                feature_aligner.eval()
            eval_accuracy = eval_batch(
                obs_enc, feature_aligner, test_demo_frames_data, test_demo_frames_label, batch_size, aligment_mode)
            # Set the model back to training mode
            if aligment_mode == 'fine_tune':
                obs_enc.train()
            if aligment_mode == 'augment':
                feature_aligner.train()
            print(f"Epoch {epoch+1} test accuracy: {eval_accuracy:.4f}")

    # Save the encoder's backbone weights if the alignment mode is fine-tune
    if aligment_mode == 'fine_tune':
        torch.save(obs_enc.backbone.state_dict(),
                   'mae_pretrain_hoi_vit_small_fine_tune.pth')
    if aligment_mode == 'augment':
        torch.save(feature_aligner.state_dict(), 'feature_aligner_'+aligner+'.pth')
else:
    # Set the model to evaluation mode
    if aligment_mode == 'fine_tune':
        obs_enc.eval()
    if aligment_mode == 'augment':
        feature_aligner.eval()


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




if aligner == 'MAP':
    feature_aligner.load_state_dict(torch.load('feature_aligner_MAP.pth'))
    eval_accuracy = eval_batch(
                obs_enc, feature_aligner, test_demo_frames_data, test_demo_frames_label, batch_size, aligment_mode)
    print(f"Test accuracy of the loaded aligner: {eval_accuracy:.4f}")

if aligner == 'TFEncoder':
    feature_aligner.load_state_dict(torch.load('feature_aligner_TFEncoder.pth'))
    print(f"Test accuracy of the loaded aligner: {eval_accuracy:.4f}")





test_image_path = '/home/thomastian/workspace/mvp/mvp_fine_tune_data/cabinet/positive/6/140.png'
test_image = Image.open(test_image_path)
test_image.show()
# Convert the image to a tensor
test_image_tensor = pixel_to_tensor(np.asarray(test_image))[0].cuda().float()
encoded_patch_embed = obs_enc.backbone.extract_encoded_patch_embed(test_image_tensor)
outputs = feature_aligner(encoded_patch_embed)
print('Positive frame prediction: ', outputs)


# Get the original observation encoder attention
obs_encoder_ave_attention = get_image_encoder_ave_attention(obs_enc, test_image_tensor)
# Save the image
output_dir = '/home/thomastian/workspace/mvp/mvp_fine_tune_data/cabinet/exp_data/encoder_ave_att_image.png'
encoder_ave_att_image = attention_to_image(obs_encoder_ave_attention, 224, 16, output_dir)


# Get the feature aligner attention
feature_aligner_attention = feature_aligner.get_feature_aligner_attention(encoded_patch_embed)
# Save the image
output_dir = '/home/thomastian/workspace/mvp/mvp_fine_tune_data/cabinet/exp_data/aligner_ave_att_image.png'
encoder_ave_att_image = attention_to_image(feature_aligner_attention, 224, 16, output_dir)




#########################################################################################################################
# # iterate over the elements of attentions
# weighted_attentions = 0
# for i in range(attentions.shape[1]):
#     current_patch_weight = attentions[:, i]
#     if current_patch_weight > 0:
#         # Find the attention of this patch in the encoder
#         current_patch_attentions = encoder_attentions[0, :, i, 1:].reshape(encoder_nh, -1)
#         # Average the attention of the patch over the heads
#         current_patch_attentions = current_patch_attentions.sum(dim=0) / encoder_nh
#         # Weight the attention of the patch by the weight of the patch
#         current_patch_attentions = current_patch_attentions * current_patch_weight
#         weighted_attentions += current_patch_attentions


# w_featmap = 224 // 16
# h_featmap = 224 // 16


# weighted_attentions = weighted_attentions.reshape(1, w_featmap, h_featmap)
# weighted_attentions = nn.functional.interpolate(weighted_attentions.unsqueeze(
#     0), scale_factor=16, mode="nearest")[0].cpu().detach().numpy()

# output_dir = '/home/thomastian/workspace/mvp/mvp_exp_data/attention_test'
# fname = os.path.join(output_dir, "new.png")
# plt.imsave(fname=fname, arr=weighted_attentions[0], format='png')
# print(f"{fname} saved.")

# # Find which patch is the most important
# print(np.argmax(attentions_patch))

    

# if aligment_mode == 'fine_tune':

#     # Run eval using the original model
#     eval_accuracy = eval_batch(
#         feature_aligner, test_demo_frames_data, test_demo_frames_label, batch_size)
#     obs_enc.eval()
#     print(f"Pre-trained model accuracy: {eval_accuracy:.4f}")

#     test_image_path = '/home/thomastian/workspace/mvp/mvp_exp_data/attention_test/148.png'

#     attentions = get_model_average_attention(test_image_path, obs_enc)

#     output_dir = '/home/thomastian/workspace/mvp/mvp_exp_data/attention_test'
#     fname = os.path.join(output_dir, "pretrained_mvp_attn-ave-heads.png")
#     plt.imsave(fname=fname, arr=attentions[0], format='png')
#     print(f"{fname} saved.")

#     # Now we load the new model and save the attention
#     obs_enc.backbone.load_state_dict(torch.load(
#         'mae_pretrain_hoi_vit_small_fine_tune.pth'))
#     obs_enc.cuda()
#     obs_enc.eval()
#     eval_accuracy = eval_batch(
#         feature_aligner, test_demo_frames_data, test_demo_frames_label, batch_size)
#     print(f"Fine-tuned model accuracy: {eval_accuracy:.4f}")

#     attentions_aligned = get_model_average_attention(test_image_path, obs_enc)

#     output_dir = '/home/thomastian/workspace/mvp/mvp_exp_data/attention_test'
#     fname = os.path.join(
#         output_dir, "fine_tuned_with_contrastive-attn-ave-head.png")
#     plt.imsave(fname=fname, arr=attentions_aligned[0], format='png')
#     print(f"{fname} saved.")

#     print(attentions - attentions_aligned)