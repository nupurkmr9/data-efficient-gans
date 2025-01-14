# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

from DiffAugment_pytorch import DiffAugment
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, diffaugment='', augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=0.):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.diffaugment = diffaugment
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.diffaugment:
            img = DiffAugment(img, policy=self.diffaugment)
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
            gen_logits = self.run_D(gen_img, gen_c, sync=False)
            loss_Gmain = torch.nn.functional.softplus(-gen_logits) 
            loss_Gmain = loss_Gmain.mean().mul(gain)
            loss_Gmain.backward()
                
            return loss_Gmain.mean()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
            gen_logits = self.run_D(gen_img, gen_c, sync=False) 
            loss_Dgen = torch.nn.functional.softplus(gen_logits) 
            loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
            real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
            
            loss_Dreal = 0
            if do_Dmain:
                loss_Dreal = torch.nn.functional.softplus(-real_logits) 

            loss_Dr1 = 0
            if do_Dr1:
                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], 
                                               create_graph=True, only_inputs=True)[0]
                r1_penalty = r1_grads.square().sum([1,2,3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)

        (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        return loss_Dreal.mean()

#----------------------------------------------------------------------------

