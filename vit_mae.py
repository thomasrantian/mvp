import numpy as np
import os
import torch
import torch.nn as nn

import torchvision
import vit2 as vit


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

vit_im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device='cpu').view(3, 1, 1)
vit_im_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device='cpu').view(3, 1, 1)

# MVP Encoder [VIT + a linear layer]
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



# Set the encoder config
encoder_cfg = {'name': 'vits-mae-hoi', 'pretrain_dir':
               '/home/thomastian/workspace//mvp_exp_data/mae_encoders', 'freeze': True, 'emb_dim': 128}

# Initialize the encoder
obs_enc = Encoder(
    model_name=encoder_cfg["name"],
    pretrain_dir=encoder_cfg["pretrain_dir"],
    freeze=encoder_cfg["freeze"],
    emb_dim=encoder_cfg["emb_dim"]
).cuda()

def pixel_to_tensor(arr, im_mean, im_std):
    '''Converts a image numpy array to a torch tensor.'''
    arr = torch.from_numpy(arr).permute(0, 3, 1, 2)
    arr = arr / 255.0
    # subtract the mean and divide by the standard deviation
    arr = (arr - im_mean) / im_std
    return arr.float()

# Run inference [input is B x 224 x 224 x images]
test_image = np.random.rand(5, 224, 224, 3)
test_image = pixel_to_tensor(test_image, vit_im_mean, vit_im_std).cuda()
test_image_embd,_ = obs_enc(test_image)

print(test_image_embd.shape)
