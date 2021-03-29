# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
import torch_xla.utils.gcsfs as gcsfs
from torch_xla._patched_functions import *

import legacy
from metrics import metric_main
import torch_xla.debug.metrics as met

import torch_xla.core.xla_model as xm

       
#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def _get_device_spec(device):
    ordinal = xm.get_ordinal(defval=-1)
    return str(device) if ordinal < 0 else '{}/{}'.format(device, ordinal)


def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 0.5,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    device = xm.xla_device()
    rank = xm.get_ordinal()
    start_time = time.time()
    save_path = 'gs://ganbucker/'
    
    stats_tfevents = None
    if xm.is_master_ordinal():
        xm.master_print('Initializing tensorboard...')
        import torch.utils.tensorboard as tensorboard
        stats_tfevents = tensorboard.SummaryWriter(run_dir)
        print(stats_tfevents)
       
    # Initialize.
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    xm.master_print('Loading training set...')
       
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = torch.utils.data.distributed.DistributedSampler(
          training_set,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
  
    training_set_loader = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, 
                                                      batch_size=batch_size//num_gpus, **data_loader_kwargs)
    training_set_parallel_loader =  pl.ParallelLoader(training_set_loader , [device])
    training_set_iterator = training_set_parallel_loader.per_device_loader(device) # iter(training_set_parallel_loader)

    # Construct networks.
    xm.master_print('Constructing networks...')
        
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) 
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) 
    
    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D)]:
            ##note not handling resume right now
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

 
    # Setup augmentation.
    xm.master_print('Setting up augmentation...')
        
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) 
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        
    # Distribute across GPUs.
      
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module = module.to(device)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    xm.master_print('Setting up training phases...')
        
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs  in [('G', G, G_opt_kwargs), ('D', D, D_opt_kwargs)]:
        opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
        phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
      

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None


    xm.master_print(f'Training for {total_kimg} kimg...')
        
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    batch_idx = 0
   

    while True:
        try:
            phase_real_img, phase_real_c = next(training_set_iterator)
        except:
            training_set_iterator =  training_set_parallel_loader.per_device_loader(device)
            phase_real_img, phase_real_c = next(training_set_iterator)
       
        phase_real_img = (phase_real_img.to(torch.float32) / 127.5 - 1).to(device)
        phase_real_c = phase_real_c.to(device)
        all_gen_z = torch.randn((2 * batch_gpu, G.z_dim), device=device).split(batch_gpu)
        all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(2 * batch_gpu)]
        all_gen_c = torch.from_numpy(np.stack(all_gen_c)).to(device)

        # Execute training phases.
        
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            # Accumulate gradients over multiple rounds.
            if rank ==0:
                sync = True
            else:
                sync = False
            gain = phase.interval
            if 'G' in phase.name:
                loss_item_g = loss.accumulate_gradients(phase=phase.name, real_img=phase_real_img, real_c=phase_real_c, 
                                      gen_z=phase_gen_z, gen_c=phase_gen_c, sync=sync, gain=gain)
            else:
                loss_item_d = loss.accumulate_gradients(phase=phase.name, real_img=phase_real_img, real_c=phase_real_c, 
                                      gen_z=phase_gen_z, gen_c=phase_gen_c, sync=sync, gain=gain)
            
            # Update weights.
            phase.module.requires_grad_(False)
            #gradient clipping sentence here
            xm.reduce_gradients(phase.opt)
            xm.master_print("grad norm clipping")
            phase.module.clipgrad()
            phase.opt.step()
#             xm.optimizer_step(phase.opt)
            xm.mark_step()
            
#         print(met.metrics_report())

        # Update G_ema.
        ema_nimg = ema_kimg * 1000
        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
        G.EMA(ema_beta)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        done = (cur_nimg >= total_kimg * 1000)
        
        if stats_tfevents is not None:
            stats_tfevents.add_scalar("loss_g", loss_item_g, global_step=int(cur_nimg / 1e3))
            stats_tfevents.add_scalar("loss_d", loss_item_d, global_step=int(cur_nimg / 1e3))
            stats_tfevents.flush()

        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            xm.master_print('Aborting...')

        
#         # Save network snapshot.
#         if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
#             for name, module in [('G', G), ('D', D)]:
#                 with gcsfs.open(save_path + str(name) + str(cur_tick) + '.pt', mode='w') as writefile:
#                     xm.save(module.state_dict(), writefile)


        def _train_update(device, cur_nimg,walltime ,loss_d, loss_g , stats_tfevents):
            global_step = int(cur_nimg / 1e3)

            update_data = [
                  'Training', 'Device={}'.format(_get_device_spec(device)),
                  'Kimg={}'.format(global_step),
                  'Time={}'.format(walltime), 'Loss g={:.5f}'.format(loss_g),
                  'Loss D={:.5f}'.format(loss_d),
              ]
            print('|', ' '.join(item for item in update_data if item), flush=True)

        cur_tick += 1
        timestamp = time.time()
        walltime = timestamp - start_time
        xm.add_step_closure(
                _train_update, args=(device, cur_nimg,walltime ,loss_item_d, loss_item_g , stats_tfevents))


    
        
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------



