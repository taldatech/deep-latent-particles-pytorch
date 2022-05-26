"""
Main training function for multi-GPU machines.
We use HuggingFace Accelerate: https://huggingface.co/docs/accelerate/index
1. Set visible GPUs under: `os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"`
2. Set "num_processes": NUM_GPUS in `accel_conf.json`

Default hyper-parameters
+---------+--------------------------------+----------+-------------+---------------+---------+------------+------------+----------+---------------------+
| dataset |        model (dec_bone)        | n_kp_enc | n_kp_prior  | rec_loss_func | beta_kl | kl_balance | patch_size | anchor_s | learned_feature_dim |
+---------+--------------------------------+----------+-------------+---------------+---------+------------+------------+----------+---------------------+
| celeb   | masked (gauss_pointnetpp_feat) |       30 |          50 | vgg           |      40 |      0.001 |          8 |    0.125 |                  10 |
| traffic | object (gauss_pointnetpp)      |       15 |          20 | vgg           |      30 |      0.001 |         16 |     0.25 |                  20 |
| clevrer | object (gauss_pointnetpp)      |       10 |          20 | vgg           |      40 |      0.001 |         16 |     0.25 |                   5 |
| shapes  | object (gauss_pointnetpp)      |        8 |          15 | mse           |    0.1 |      0.001 |          8 |     0.25 |                   5 |
+---------+--------------------------------+----------+-------------+---------------+---------+------------+------------+----------+---------------------+
"""
# imports
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"  # "0, 1, 2, 3"
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import argparse
# torch
import torch
import torch.nn.functional as F
from utils.loss_functions import ChamferLossKL, calc_kl, calc_reconstruction_loss, VGGDistance
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
# modules
from models import KeyPointVAE
# datasets
from datasets.celeba_dataset import CelebAPrunedAligned_MAFLVal, evaluate_lin_reg_on_mafl
from datasets.traffic_ds import TrafficDataset
from datasets.clevrer_ds import CLEVRERDataset
from datasets.shapes_ds import generate_shape_dataset_torch
# util functions
from utils.util_func import plot_keypoints_on_image_batch, create_masks_fast, prepare_logdir, \
    save_config, log_line, plot_bb_on_image_batch_from_masks_nms
from eval.eval_model import evaluate_validation_elbo
from accelerate import Accelerator, DistributedDataParallelKwargs

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_dlp(ds="celeba", batch_size=16, lr=5e-4, kp_activation="none",
              pad_mode='replicate', num_epochs=250, load_model=False, n_kp=8, recon_loss_type="mse",
              use_logsoftmax=False, sigma=0.1, beta_kl=1.0, beta_rec=1.0, dropout=0.0, dec_bone="gauss",
              patch_size=16, topk=15, n_kp_enc=20, eval_epoch_freq=5,
              learned_feature_dim=0, n_kp_prior=100, weight_decay=0.0, kp_range=(0, 1),
              run_prefix="", mask_threshold=0.2, use_tps=False, use_pairs=False, use_object_enc=True,
              use_object_dec=False, warmup_epoch=5, iou_thresh=0.2, anchor_s=0.25, learn_order=False,
              kl_balance=0.1, exclusive_patches=False):
    """
    ds: dataset name (str)
    enc_channels: channels for the posterior CNN (takes in the whole image)
    prior_channels: channels for prior CNN (takes in patches)
    n_kp: number of kp to extract from each (!) patch
    n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
    n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
    use_logsoftmax: for spatial-softmax, set True to use log-softmax for numerical stability
    pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
    sigma: the prior std of the KP
    dropout: dropout for the CNNs. We don't use it though...
    dec_bone: decoder backbone -- "gauss_pointnetpp_feat": Masked Model, "gauss_pointnetpp": Object Model
    patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
    kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
    learned_feature_dim: the latent visual features dimensions extracted from glimpses.
    kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
    mask_threshold: activation threshold (>thresh -> 1, else 0) for the binary mask created from the Gaussian-maps.
    anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
    learn_order: experimental feature to learn the order of keypoints - but it doesn't work yet.
    use_object_enc: set True to use a separate encoder to encode visual features of glimpses.
    use_object_dec: set True to use a separate decoder to decode glimpses (Object Model).
    iou_thresh: intersection-over-union threshold for non-maximal suppression (nms) to filter bounding boxes
    use_tps: set True to use a tps augmentation on the input image for datasets that support this option
    use_pairs: for CelebA dataset, set True to use a tps-augmented image for the prior.
    topk: the number top-k particles with the lowest variance (highest confidence) to filter for the plots.
    warmup_epoch: (used for the Object Model) number of epochs where only the object decoder is trained.
    recon_loss_type: tpe of pixel reconstruction loss ("mse", "vgg").
    beta_rec: coefficient for the reconstruction loss (we use 1.0).
    beta_kl: coefficient for the KL divergence term in the loss.
    kl_balance: coefficient for the balance between the ChamferKL (for the KP)
                and the standard KL (for the visual features),
                kl_loss = beta_kl * (chamfer_kl + kl_balance * kl_features)
    exclusive_patches: (mostly) enforce one particle pre object by masking up regions that were already encoded.
    """
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # in conf: "num_processes": num_visible_gpus
    # load data
    if ds == "celeba":
        image_size = 128
        imwidth = 160
        crop = 16
        ch = 3
        # enc_channels = [64, 128, 256, 512]
        enc_channels = [32, 64, 128, 256]
        # prior_channels = (16, 16, 32)
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/celebaa'
        if use_tps:
            import tps
            # warper = tps.WarperSingle(H=imwidth, W=imwidth)
            if use_pairs:
                warper = tps.Warper(H=imwidth, W=imwidth, im1_multiplier=0.1, im1_multiplier_aff=0.1)
            else:
                warper = tps.WarperSingle(H=imwidth, W=imwidth, warpsd_all=0.001, warpsd_subset=0.01, transsd=0.1,
                                          scalesd=0.1, rotsd=5)
            print('using tps augmentation')
        else:
            warper = None
        dataset = CelebAPrunedAligned_MAFLVal(root=root, train=True, do_augmentations=False, imwidth=imwidth, crop=crop,
                                              pair_warper=warper)
        # milestones = (30, 60, 90, 120)
        milestones = (50, 80, 100)
    elif ds == "traffic":
        image_size = 128
        ch = 3
        # enc_channels = [64, 128, 256, 512]
        enc_channels = [32, 64, 128, 256]
        # prior_channels = (16, 16, 32)
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/traffic_dataset/img128np_fs3.npy'
        mode = 'single'
        dataset = TrafficDataset(path_to_npy=root, image_size=image_size, mode=mode, train=True)
        milestones = (50, 80, 100)
    elif ds == 'clevrer':
        image_size = 128
        ch = 3
        # enc_channels = [64, 128, 256, 512]
        enc_channels = [32, 64, 128, 256]
        # prior_channels = (16, 16, 32)
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/clevrer/clevrer_img128np_fs3_train.npy'
        # root = '/media/newhd/data/clevrer/valid/clevrer_img128np_fs3_valid.npy'
        mode = 'single'
        dataset = CLEVRERDataset(path_to_npy=root, image_size=image_size, mode=mode, train=True)
        milestones = (30, 60, 100)
    elif ds == "shapes":
        image_size = 64
        ch = 3
        enc_channels = [32, 64, 128]
        prior_channels = (16, 32, 64)
        print('generating random shapes dataset')
        dataset = generate_shape_dataset_torch(num_images=20_000)
        milestones = (20, 40, 80)
    else:
        raise NotImplementedError

    hparams = {'ds': ds, 'batch_size': batch_size, 'lr': lr, 'kp_activation': kp_activation, 'pad_mode': pad_mode,
               'num_epochs': num_epochs, 'n_kp': n_kp, 'recon_loss_type': recon_loss_type,
               'use_logsoftmax': use_logsoftmax, 'sigma': sigma, 'beta_kl': beta_kl, 'beta_rec': beta_rec,
               'dec_bone': dec_bone, 'patch_size': patch_size, 'topk': topk, 'n_kp_enc': n_kp_enc,
               'eval_epoch_freq': eval_epoch_freq, 'learned_feature_dim': learned_feature_dim,
               'n_kp_prior': n_kp_prior, 'weight_decay': weight_decay, 'kp_range': kp_range,
               'run_prefix': run_prefix, 'mask_threshold': mask_threshold, 'use_tps': use_tps, 'use_pairs': use_pairs,
               'use_object_enc': use_object_enc, 'use_object_dec': use_object_dec, 'warmup_epoch': warmup_epoch,
               'iou_thresh': iou_thresh, 'anchor_s': anchor_s, 'learn_order': learn_order, 'kl_balance': kl_balance,
               'milestones': milestones, 'image_size': image_size, 'enc_channels': enc_channels,
               'prior_channels': prior_channels, 'exclusive_patches': exclusive_patches}

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True,
                            drop_last=True)
    # model
    model = KeyPointVAE(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                        image_size=image_size, n_kp=n_kp, learned_feature_dim=learned_feature_dim,
                        use_logsoftmax=use_logsoftmax, pad_mode=pad_mode, sigma=sigma,
                        dropout=dropout, dec_bone=dec_bone, patch_size=patch_size, n_kp_enc=n_kp_enc,
                        n_kp_prior=n_kp_prior, kp_range=kp_range, kp_activation=kp_activation,
                        mask_threshold=mask_threshold, use_object_enc=use_object_enc, use_object_dec=use_object_dec,
                        anchor_s=anchor_s, learn_order=learn_order, exclusive_patches=exclusive_patches)
    logvar_p = torch.log(torch.tensor(sigma ** 2)).to(accelerator.device)  # logvar of the constant std -> for the kl
    # prepare saving location
    run_name = f'{ds}_dlp_{dec_bone}' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')
    save_config(log_dir, hparams)

    kl_loss_func = ChamferLossKL(use_reverse_kl=False)
    if recon_loss_type == "vgg":
        recon_loss_func = VGGDistance(device=accelerator.device)
    else:
        recon_loss_func = calc_reconstruction_loss
    betas = (0.9, 0.999)
    eps = 1e-4
    optimizer_e = optim.Adam(model.get_parameters(encoder=True, prior=True, decoder=False), lr=lr, betas=betas, eps=eps,
                             weight_decay=weight_decay)
    optimizer_d = optim.Adam(model.get_parameters(encoder=False, prior=False, decoder=True), lr=lr, betas=betas,
                             eps=eps, weight_decay=weight_decay)

    # convert BatchNorm to SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model, optimizer_e, optimizer_d, dataloader = accelerator.prepare(model, optimizer_e, optimizer_d, dataloader)

    scheduler_e = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=milestones, gamma=0.1)
    scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=milestones, gamma=0.1)

    if load_model:
        try:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(
                torch.load(os.path.join(save_dir, f'{ds}_dlp_{dec_bone}.pth'),
                           map_location=accelerator.device))
            print("loaded model from checkpoint")
        except:
            print("model checkpoint not found")

    losses = []
    losses_rec = []
    losses_kl = []
    losses_kl_kp = []
    losses_kl_feat = []

    linreg_error = best_linreg_error = 1.0
    best_linreg_epoch = 0
    linreg_logvar_error = best_linreg_logvar_error = 1.0
    best_linreg_logvar_epoch = 0
    linreg_features_error = best_linreg_features_error = 1.0
    best_linreg_features_epoch = 0

    linreg_errors = []
    linreg_logvar_errors = []
    linreg_features_errors = []

    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        batch_losses_rec = []
        batch_losses_kl = []
        batch_losses_kl_kp = []
        batch_losses_kl_feat = []

        pbar = tqdm(iterable=dataloader, disable=not accelerator.is_local_main_process)
        for batch in pbar:
            if ds == 'playground':
                prev_obs, obs = batch[0][:, 0], batch[0][:, 1]
                # prev_obs, obs = prev_obs.to(device), obs.to(device)
                x = prev_obs.to(accelerator.device)
                x[:, 1][x[:, 1] == 0.0] = 0.4
                x_prior = x
            elif ds == 'celeba':
                if len(batch['data'].shape) == 5:
                    x_prior = batch['data'][:, 0].to(accelerator.device)
                    x = batch['data'][:, 1].to(accelerator.device)
                else:
                    x = batch['data'].to(accelerator.device)
                    x_prior = x
            elif ds == 'replay_buffer':
                x = batch[0].to(accelerator.device)
                x_prior = x
            elif ds == 'traffic':
                if mode == 'single':
                    x = batch.to(accelerator.device)
                    # x = normalize(x)
                    x_prior = x
                else:
                    x = batch[0].to(accelerator.device)
                    # x = normalize(x)
                    x_prior = batch[1].to(accelerator.device)
                    # x_prior = normalize(x_prior)
            elif ds == 'bair':
                x = batch[:, 1].to(accelerator.device)
                x_prior = batch[:, 0].to(accelerator.device)
            elif ds == 'clevrer':
                if mode == 'single':
                    x = batch.to(accelerator.device)
                    x_prior = x
                else:
                    x = batch[0].to(accelerator.device)
                    x_prior = batch[1].to(accelerator.device)
            else:
                x = batch
                x_prior = x
            batch_size = x.shape[0]
            # forward pass
            use_stg = False
            noisy_masks = (epoch < 5 * warmup_epoch)
            model_output = model(x, x_prior=x_prior, warmup=(epoch < warmup_epoch), stg=use_stg,
                                 noisy_masks=noisy_masks)
            mu_p = model_output['kp_p']
            gmap = model_output['gmap']
            mu = model_output['mu']
            logvar = model_output['logvar']
            rec_x = model_output['rec']
            mu_features = model_output['mu_features']
            logvar_features = model_output['logvar_features']
            # object stuff
            dec_objects_original = model_output['dec_objects_original']
            cropped_objects_original = model_output['cropped_objects_original']
            obj_on = model_output['obj_on']  # [batch_size, n_kp]
            # reconstruction error
            if use_object_dec and dec_objects_original is not None and epoch < warmup_epoch:
                if recon_loss_type == "vgg":
                    _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                    dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                    cropped_objects_original = cropped_objects_original.reshape(-1,
                                                                                *cropped_objects_original.shape[2:])
                    if cropped_objects_original.shape[-1] < 32:
                        cropped_objects_original = F.interpolate(cropped_objects_original, size=32, mode='bilinear',
                                                                 align_corners=False)
                        dec_objects_rgb = F.interpolate(dec_objects_rgb, size=32, mode='bilinear',
                                                        align_corners=False)
                    loss_rec_obj = recon_loss_func(cropped_objects_original, dec_objects_rgb, reduction="mean")

                else:
                    _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                    dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                    cropped_objects_original = cropped_objects_original.clone().reshape(-1,
                                                                                        *cropped_objects_original.shape[
                                                                                         2:])
                    loss_rec_obj = calc_reconstruction_loss(cropped_objects_original, dec_objects_rgb,
                                                            loss_type='mse', reduction='mean')
                loss_rec = loss_rec_obj + (0 * rec_x).mean()  # + (0 * rec_x).mean() for distributed training
            else:
                if recon_loss_type == "vgg":
                    loss_rec = recon_loss_func(x, rec_x, reduction="mean")
                else:
                    loss_rec = calc_reconstruction_loss(x, rec_x, loss_type='mse', reduction='mean')

            # kl-divergence
            logvar_kp = logvar_p.expand_as(mu_p)
            # the final kp is the bg kp which is located in the center (so no need for it)
            # to reproduce the results on celeba, use `mu_post = mu`, `logvar_post = logvar`
            mu_post = mu[:, :-1]
            logvar_post = logvar[:, :-1]
            mu_prior = mu_p
            logvar_prior = logvar_kp

            loss_kl_kp = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
                                      logvar_gts=logvar_prior).mean()

            if learned_feature_dim > 0:
                loss_kl_feat = calc_kl(logvar_features.view(-1, logvar_features.shape[-1]),
                                       mu_features.view(-1, mu_features.shape[-1]), reduce='none')
                loss_kl_feat = loss_kl_feat.view(batch_size, n_kp_enc + 1).sum(1).mean()
            else:
                loss_kl_feat = torch.tensor(0.0, device=accelerator.device)

            loss_kl = loss_kl_kp + kl_balance * loss_kl_feat

            loss = beta_rec * loss_rec + beta_kl * loss_kl
            # backprop
            optimizer_e.zero_grad()
            optimizer_d.zero_grad()
            accelerator.backward(loss)
            optimizer_e.step()
            optimizer_d.step()
            # log
            batch_losses.append(loss.data.cpu().item())
            batch_losses_rec.append(loss_rec.data.cpu().item())
            batch_losses_kl.append(loss_kl.data.cpu().item())
            batch_losses_kl_kp.append(loss_kl_kp.data.cpu().item())
            batch_losses_kl_feat.append(loss_kl_feat.data.cpu().item())
            # progress bar
            if use_object_dec and epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            elif use_object_dec and noisy_masks:
                pbar.set_description_str(f'epoch #{epoch} (noisy masks)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')
            # pbar.set_description_str('epoch #{}'.format(epoch))
            pbar.set_postfix(loss=loss.data.cpu().item(), rec=loss_rec.data.cpu().item(),
                             kl=loss_kl.data.cpu().item())
        pbar.close()
        losses.append(np.mean(batch_losses))
        losses_rec.append(np.mean(batch_losses_rec))
        losses_kl.append(np.mean(batch_losses_kl))
        losses_kl_kp.append(np.mean(batch_losses_kl_kp))
        losses_kl_feat.append(np.mean(batch_losses_kl_feat))
        # keep track of bounding box scores to set a hard threshold (as bb scores are not normalized)
        # epoch_bb_scores = torch.cat(batch_bb_scores, dim=0)
        # bb_mean_score = epoch_bb_scores.mean().data.cpu().item()
        # bb_mean_scores.append(bb_mean_score)
        # schedulers
        scheduler_e.step()
        scheduler_d.step()
        # epoch summary
        log_str = f'epoch {epoch} summary for dec backbone: {dec_bone}\n'
        log_str += f'loss: {losses[-1]:.3f}, rec: {losses_rec[-1]:.3f}, kl: {losses_kl[-1]:.3f}\n'
        log_str += f'kl_balance: {kl_balance:.3f}, kl_kp: {losses_kl_kp[-1]:.3f}, kl_feat: {losses_kl_feat[-1]:.3f}\n'
        log_str += f'mu max: {mu.max()}, mu min: {mu.min()}\n'
        if ds != 'celeba':
            log_str += f'val loss (freq: {eval_epoch_freq}): {valid_loss:.3f},' \
                       f' best: {best_valid_loss:.3f} @ epoch: {best_valid_epoch}\n'
        if obj_on is not None:
            log_str += f'obj_on max: {obj_on.max()}, obj_on min: {obj_on.min()}\n'
        accelerator.print(log_str)
        if accelerator.is_main_process:
            log_line(log_dir, log_str)
        # wait an unwrap model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            if accelerator.is_main_process:
                max_imgs = 8
                img_with_kp = plot_keypoints_on_image_batch(mu[:, :-1].clamp(min=kp_range[0], max=kp_range[1]), x,
                                                            radius=3, thickness=1, max_imgs=max_imgs, kp_range=kp_range)
                img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x_prior, radius=3, thickness=1, max_imgs=max_imgs,
                                                              kp_range=kp_range)
                # top-k
                with torch.no_grad():
                    logvar_sum = logvar[:, :-1].sum(-1)
                    logvar_topk = torch.topk(logvar_sum, k=topk, dim=-1, largest=False)
                    indices = logvar_topk[1]  # [batch_size, topk]
                    batch_indices = torch.arange(mu.shape[0]).view(-1, 1).to(mu.device)
                    topk_kp = mu[batch_indices, indices]
                    # bounding boxes
                    masks = create_masks_fast(mu[:, :-1].detach(), anchor_s=unwrapped_model.anchor_s,
                                              feature_dim=x.shape[-1])
                    masks = torch.where(masks < mask_threshold, 0.0, 1.0)
                    bb_scores = -1 * logvar_sum
                    hard_threshold = bb_scores.mean()
                if use_object_dec:
                    img_with_masks_nms, nms_ind = plot_bb_on_image_batch_from_masks_nms(masks, x, scores=bb_scores,
                                                                                        iou_thresh=iou_thresh,
                                                                                        thickness=1, max_imgs=max_imgs,
                                                                                        hard_thresh=hard_threshold)
                    # hard_thresh: a general threshold for bb scores (set None to not use it)
                    bb_str = f'bb scores: max: {bb_scores.max():.2f}, min: {bb_scores.min():.2f},' \
                             f' mean: {bb_scores.mean():.2f}\n'
                    accelerator.print(bb_str)
                    log_line(log_dir, bb_str)
                img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                                 radius=3, thickness=1, max_imgs=max_imgs,
                                                                 kp_range=kp_range)

                if use_object_dec and dec_objects_original is not None:
                    dec_objects = model_output['dec_objects']
                    vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(accelerator.device),
                                                 rec_x[:max_imgs, -3:],
                                                 img_with_kp_p[:max_imgs, -3:].to(accelerator.device),
                                                 img_with_kp_topk[:max_imgs, -3:].to(accelerator.device),
                                                 dec_objects[:max_imgs, -3:],
                                                 img_with_masks_nms[:max_imgs, -3:].to(accelerator.device)],
                                                dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
                                      nrow=8, pad_value=1)
                    with torch.no_grad():
                        _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                        dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                        cropped_objects_original = cropped_objects_original.clone().reshape(-1, 3,
                                                                                            cropped_objects_original.shape[
                                                                                                -1],
                                                                                            cropped_objects_original.shape[
                                                                                                -1])
                        if cropped_objects_original.shape[-1] != dec_objects_rgb.shape[-1]:
                            cropped_objects_original = F.interpolate(cropped_objects_original,
                                                                     size=dec_objects_rgb.shape[-1],
                                                                     align_corners=False, mode='bilinear')
                    vutils.save_image(
                        torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                                  dim=0).data.cpu(), '{}/image_obj_{}.jpg'.format(fig_dir, epoch),
                        nrow=8, pad_value=1)
                else:
                    vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(accelerator.device),
                                                 rec_x[:max_imgs, -3:],
                                                 img_with_kp_p[:max_imgs, -3:].to(accelerator.device),
                                                 img_with_kp_topk[:max_imgs, -3:].to(accelerator.device)],
                                                dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
                                      nrow=8, pad_value=1)
                accelerator.save(unwrapped_model.state_dict(),
                                 os.path.join(save_dir, f'{ds}_dlp_{dec_bone}{run_prefix}.pth'))
            eval_model = unwrapped_model
            if ds == "celeba":
                if accelerator.is_main_process:
                    # evaluate supervised linear regression errors
                    accelerator.print("evaluating linear regression error...")
                    linreg_error_train, linreg_error = evaluate_lin_reg_on_mafl(eval_model, root=root, use_logvar=False,
                                                                                batch_size=100,
                                                                                device=accelerator.device,
                                                                                img_size=image_size,
                                                                                fig_dir=fig_dir,
                                                                                epoch=epoch)
                    if best_linreg_error > linreg_error:
                        best_linreg_error = linreg_error
                        best_linreg_epoch = epoch
                    linreg_logvar_error_train, linreg_logvar_error = evaluate_lin_reg_on_mafl(eval_model, root=root,
                                                                                              use_logvar=True,
                                                                                              batch_size=100,
                                                                                              device=accelerator.device,
                                                                                              img_size=image_size,
                                                                                              fig_dir=fig_dir,
                                                                                              epoch=epoch)
                    if best_linreg_logvar_error > linreg_logvar_error:
                        best_linreg_logvar_error = linreg_logvar_error
                        best_linreg_logvar_epoch = epoch
                    if learned_feature_dim > 0:
                        linreg_features_error_train, linreg_features_error = evaluate_lin_reg_on_mafl(eval_model,
                                                                                                      root=root,
                                                                                                      use_logvar=True,
                                                                                                      batch_size=100,
                                                                                                      device=accelerator.device,
                                                                                                      img_size=image_size,
                                                                                                      fig_dir=fig_dir,
                                                                                                      epoch=epoch,
                                                                                                      use_features=True)
                        if best_linreg_features_error > linreg_features_error:
                            best_linreg_features_error = linreg_features_error
                            best_linreg_features_epoch = epoch
                            accelerator.save(unwrapped_model.state_dict(),
                                             os.path.join(save_dir,
                                                          f'{ds}_dlp_{dec_bone}{run_prefix}_best.pth'))
                    linreg_str = f'eval epoch {epoch}: error: {linreg_error * 100:.4f}%,' \
                                 f' error with logvar: {linreg_logvar_error * 100:.4f},' \
                                 f' train logvar error: {linreg_logvar_error_train * 100:.4f}%\n'
                    # accelerator.print(
                    #     f'eval epoch {epoch}: error: {linreg_error * 100:.4f}%,'
                    #     f' error with logvar: {linreg_logvar_error * 100:.4f}%'
                    #     f' train logvar error: {linreg_logvar_error_train * 100:.4f}')
                    if learned_feature_dim > 0 and "pointnet" in dec_bone:
                        linreg_str += f'error with features: {linreg_features_error * 100:.4f}%,' \
                                      f' train logvar error: {linreg_features_error_train * 100:.4f}%\n'
                        # accelerator.print(f'error with features: {linreg_features_error * 100:.4f}% '
                        #                   f'train logvar error: {linreg_features_error_train * 100:.4f}%')
                    # accelerator.print(
                    #     f'best error {best_linreg_epoch}: {best_linreg_error * 100:.4f}%,'
                    #     f' error with logvar {best_linreg_logvar_epoch}: {best_linreg_logvar_error * 100:.4f}%')
                    linreg_str += f'best error {best_linreg_epoch}: {best_linreg_error * 100:.4f}%,' \
                                  f'  error with logvar {best_linreg_logvar_epoch}: {best_linreg_logvar_error * 100:.4f}%\n'
                    if learned_feature_dim > 0 and "pointnet" in dec_bone:
                        linreg_str += f'error with features' \
                                      f' {best_linreg_features_epoch}: {best_linreg_features_error * 100:.4f}%\n'
                        # accelerator.print(
                        #     f'error with features {best_linreg_features_epoch}: {best_linreg_features_error * 100:.4f}%')
                    accelerator.print(linreg_str)
                    log_line(log_dir, linreg_str)
            else:
                accelerator.print("validation step...")
                valid_loss = evaluate_validation_elbo(eval_model, ds, epoch, batch_size=batch_size,
                                                      recon_loss_type=recon_loss_type, device=accelerator.device,
                                                      save_image=True, fig_dir=fig_dir, topk=topk,
                                                      recon_loss_func=recon_loss_func, beta_rec=beta_rec,
                                                      beta_kl=beta_kl, kl_balance=kl_balance, accelerator=accelerator)
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    best_valid_epoch = epoch
                    accelerator.save(unwrapped_model.state_dict(),
                                     os.path.join(save_dir, f'{ds}_dlp_{dec_bone}{run_prefix}_best.pth'))

        linreg_errors.append(linreg_error * 100)
        linreg_logvar_errors.append(linreg_logvar_error * 100)
        linreg_features_errors.append(linreg_features_error * 100)
        valid_losses.append(valid_loss)
        # plot graphs
        if epoch > 0 and accelerator.is_main_process:
            num_plots = 4
            fig = plt.figure()
            ax = fig.add_subplot(num_plots, 1, 1)
            ax.plot(np.arange(len(losses[1:])), losses[1:], label="loss")
            ax.set_title(run_name)
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 2)
            ax.plot(np.arange(len(losses_kl[1:])), losses_kl[1:], label="kl", color='red')
            if learned_feature_dim > 0:
                ax.plot(np.arange(len(losses_kl_kp[1:])), losses_kl_kp[1:], label="kl_kp", color='cyan')
                ax.plot(np.arange(len(losses_kl_feat[1:])), losses_kl_feat[1:], label="kl_feat", color='green')
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 3)
            ax.plot(np.arange(len(losses_rec[1:])), losses_rec[1:], label="rec", color='green')
            ax.legend()

            if ds == 'celeba':
                ax = fig.add_subplot(num_plots, 1, 4)
                ax.plot(np.arange(len(linreg_errors[1:])), linreg_errors[1:], label="linreg_err %")
                ax.plot(np.arange(len(linreg_logvar_errors[1:])), linreg_logvar_errors[1:], label="linreg_v_err %")
                if learned_feature_dim > 0:
                    ax.plot(np.arange(len(linreg_features_errors[1:])), linreg_features_errors[1:],
                            label="linreg_f_err %")
                ax.legend()
            else:
                ax = fig.add_subplot(num_plots, 1, 4)
                ax.plot(np.arange(len(valid_losses[1:])), valid_losses[1:], label="valid_loss", color='magenta')
                ax.legend()
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/{run_name}_graph.jpg')
            plt.close('all')

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    return unwrapped_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLP Single-GPU Training")
    parser.add_argument("-d", "--dataset", type=str, default='celeba',
                        help="dataset of to train the model on: ['celeba', 'traffic', 'clevrer', 'shapes']")
    parser.add_argument("-o", "--override", action='store_true',
                        help="set True to override default hyper-parameters via command line")
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=2e-4)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=100)
    parser.add_argument("-e", "--eval_freq", type=int, help="evaluation epoch frequency", default=2)
    parser.add_argument("-s", "--sigma", type=float, help="the prior std of the KP", default=0.1)
    parser.add_argument("-p", "--prefix", type=str, help="string prefix for logging", default="")
    parser.add_argument("-r", "--beta_rec", type=float, help="beta coefficient for the reconstruction loss",
                        default=1.0)
    parser.add_argument("-k", "--beta_kl", type=float, help="beta coefficient for the kl divergence",
                        default=1.0)
    parser.add_argument("-c", "--kl_balance", type=float,
                        help="coefficient for the balance between the ChamferKL (for the KP) and the standard KL",
                        default=0.001)
    parser.add_argument("-v", "--rec_loss_function", type=str, help="type of reconstruction loss: 'mse', 'vgg'",
                        default="mse")
    parser.add_argument("--n_kp_enc", type=int, help="number of posterior kp to be learned", default=30)
    parser.add_argument("--n_kp_prior", type=int, help="number of kp to filter from the set of prior kp", default=50)
    parser.add_argument("--dec_bone", type=str,
                        help="decoder backbone:'gauss_pointnetpp_feat': Masked Model, 'gauss_pointnetpp': Object Model",
                        default="gauss_pointnetpp")
    parser.add_argument("--patch_size", type=int,
                        help="patch size for the prior KP proposals network (not to be confused with the glimpse size)",
                        default=8)
    parser.add_argument("--learned_feature_dim", type=int,
                        help="the latent visual features dimensions extracted from glimpses",
                        default=10)
    parser.add_argument("--use_object_enc", action='store_true',
                        help="set True to use a separate encoder to encode visual features of glimpses")
    parser.add_argument("--use_object_dec", action='store_true',
                        help="set True to use a separate decoder to decode glimpses (Object Model)")
    parser.add_argument("--warmup_epoch", type=int,
                        help="number of epochs where only the object decoder is trained",
                        default=2)
    parser.add_argument("--anchor_s", type=float,
                        help="defines the glimpse size as a ratio of image_size", default=0.25)
    parser.add_argument("--exclusive_patches", action='store_true',
                        help="set True to enable non-overlapping object patches")
    args = parser.parse_args()

    # default hyper-parameters
    lr = 2e-4
    batch_size = 32
    num_epochs = 100
    load_model = False
    eval_epoch_freq = 2
    n_kp = 1  # num kp per patch
    mask_threshold = 0.2  # mask threshold for the features from the encoder
    kp_range = (-1, 1)
    weight_decay = 0.0
    run_prefix = ""
    learn_order = False
    use_logsoftmax = False
    pad_mode = 'replicate'
    sigma = 0.1  # default sigma for the gaussian maps
    dropout = 0.0
    kp_activation = "tanh"

    # dataset specific
    ds = args.dataset
    if args.dataset == 'celeb':
        beta_kl = 40.0
        beta_rec = 1.0
        n_kp_enc = 30  # total kp to output from the encoder / filter from prior
        n_kp_prior = 50
        patch_size = 8
        learned_feature_dim = 10  # additional features than x,y for each kp
        dec_bone = "gauss_pointnetpp_feat"
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "vgg"
        use_tps = True
        use_pairs = True
        use_object_enc = True  # separate object encoder
        use_object_dec = False  # separate object decoder
        warmup_epoch = 0
        anchor_s = 0.125
        kl_balance = 0.001
        exclusive_patches = False
    elif args.dataset == 'traffic':
        beta_kl = 30.0
        beta_rec = 1.0
        n_kp_enc = 15  # total kp to output from the encoder / filter from prior
        n_kp_prior = 20
        patch_size = 16
        learned_feature_dim = 10  # additional features than x,y for each kp
        dec_bone = "gauss_pointnetpp"
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "vgg"
        use_tps = False
        use_pairs = False
        use_object_enc = True  # separate object encoder
        use_object_dec = True  # separate object decoder
        warmup_epoch = 2
        anchor_s = 0.25
        kl_balance = 0.001
        exclusive_patches = False
    elif args.dataset == 'clevrer':
        beta_kl = 40.0
        beta_rec = 1.0
        n_kp_enc = 10  # total kp to output from the encoder / filter from prior
        n_kp_prior = 20
        patch_size = 16
        learned_feature_dim = 5  # additional features than x,y for each kp
        dec_bone = "gauss_pointnetpp"
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "vgg"
        use_tps = False
        use_pairs = False
        use_object_enc = True  # separate object encoder
        use_object_dec = True  # separate object decoder
        warmup_epoch = 1
        anchor_s = 0.25
        kl_balance = 0.001
        exclusive_patches = False
    elif args.dataset == 'shapes':
        beta_kl = 0.01
        beta_rec = 1.0
        n_kp_enc = 8  # total kp to output from the encoder / filter from prior
        n_kp_prior = 15
        patch_size = 8
        learned_feature_dim = 5  # additional features than x,y for each kp
        dec_bone = "gauss_pointnetpp"
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "mse"
        use_tps = False
        use_pairs = False
        use_object_enc = True  # separate object encoder
        use_object_dec = True  # separate object decoder
        warmup_epoch = 2
        anchor_s = 0.25
        kl_balance = 0.001
        exclusive_patches = True
        # override manually
        lr = 1e-3
        batch_size = 64
    else:
        raise NotImplementedError("unrecognized dataset, please implement it and add it to the trian script")

    override_hp = args.override
    if override_hp:
        lr = args.lr
        batch_size = args.batch_size
        num_epochs = args.num_epochs
        eval_epoch_freq = args.eval_freq
        run_prefix = args.prefix
        sigma = args.sigma
        beta_kl = args.beta_kl
        beta_rec = args.beta_rec
        n_kp_enc = args.n_kp_enc
        n_kp_prior = args.n_kp_prior
        patch_size = args.patch_size
        learned_feature_dim = args.learned_feature_dim
        dec_bone = args.dec_bone
        recon_loss_type = args.rec_loss_function
        use_object_enc = args.use_object_enc
        use_object_dec = args.use_object_dec
        warmup_epoch = args.warmup_epoch
        anchor_s = args.anchor_s
        kl_balance = args.kl_balance
        exclusive_patches = args.exclusive_patches

    model = train_dlp(ds=ds, batch_size=batch_size, lr=lr,
                      num_epochs=num_epochs, kp_activation=kp_activation,
                      load_model=load_model, n_kp=n_kp, use_logsoftmax=use_logsoftmax, pad_mode=pad_mode,
                      sigma=sigma, beta_kl=beta_kl, beta_rec=beta_rec, dropout=dropout, dec_bone=dec_bone,
                      kp_range=kp_range, learned_feature_dim=learned_feature_dim, weight_decay=weight_decay,
                      recon_loss_type=recon_loss_type, patch_size=patch_size, topk=topk, n_kp_enc=n_kp_enc,
                      eval_epoch_freq=eval_epoch_freq, n_kp_prior=n_kp_prior, run_prefix=run_prefix,
                      mask_threshold=mask_threshold, use_tps=use_tps, use_pairs=use_pairs, anchor_s=anchor_s,
                      use_object_enc=use_object_enc, use_object_dec=use_object_dec, exclusive_patches=exclusive_patches,
                      warmup_epoch=warmup_epoch, learn_order=learn_order, kl_balance=kl_balance)

    # for b_kl in [40.0, 80.0, 100.0, 200.0]:
    #     model = train_dlp(ds=ds, batch_size=batch_size, lr=lr,
    #                                 num_epochs=num_epochs, kp_activation=kp_activation,
    #                                 load_model=load_model, n_kp=n_kp, use_logsoftmax=use_logsoftmax, pad_mode=pad_mode,
    #                                 sigma=sigma, beta_kl=b_kl, beta_rec=beta_rec, dropout=dropout, dec_bone=dec_bone,
    #                                 kp_range=kp_range, learned_feature_dim=learned_feature_dim,
    #                                 weight_decay=weight_decay,
    #                                 recon_loss_type=recon_loss_type, patch_size=patch_size, topk=topk,
    #                                 n_kp_enc=n_kp_enc,
    #                                 eval_epoch_freq=eval_epoch_freq, n_kp_prior=n_kp_prior, run_prefix=run_prefix,
    #                                 mask_threshold=mask_threshold, use_tps=use_tps, use_pairs=use_pairs,
    #                                 anchor_s=anchor_s,
    #                                 use_object_enc=use_object_enc, use_object_dec=use_object_dec,
    #                                 warmup_epoch=warmup_epoch, learn_order=learn_order, kl_balance=kl_balance)
