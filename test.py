#!/usr/bin/env python3

import numpy as np
import os
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

import vit2 as vit

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
    '/home/thomastian/workspace/mvp', 'freeze': True, 'emb_dim': 64}

emb_dim = encoder_cfg["emb_dim"]

obs_enc = Encoder(
            model_name=encoder_cfg["name"],
            pretrain_dir=encoder_cfg["pretrain_dir"],
            freeze=encoder_cfg["freeze"],
            emb_dim=emb_dim
        )
print(obs_enc)
obs_emb, obs_feat = obs_enc(torch.randn(1, 3, 224, 224))
print(obs_emb.shape)