# imports
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"  # "0, 1, 2, 3"
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import cv2
# torch
import torch
import torch.nn.functional as F
from loss_functions import ChamferLossKL, calc_kl, calc_reconstruction_loss, VGGDistance
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
# modules
from models import KeyPointVAE
# from models import KeyPointVAEB as KeyPointVAE
# datasets
from celeba_dataset import CelebAPrunedAligned_MAFLVal, evaluate_lin_reg_on_mafl
from replay_buffer_dataset import ReplayBufferDataset
from traffic_ds import TrafficDataset
from playground_dataset import PlaygroundTrajectoryDataset
from bair_ds import BAIRDataset
from clevrer_ds import CLEVRERDataset
# util functions
from util_func import reparameterize, plot_keypoints_on_image_batch, create_masks_fast, prepare_logdir, save_config, \
    log_line
from eval_model import evaluate_validation_elbo
from accelerate import Accelerator, DistributedDataParallelKwargs

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_var_particles(ds="playground", batch_size=16, lr=5e-4, kp_activation="none",
                        pad_mode='zeros', num_epochs=250, load_model=False, n_kp=8, recon_loss_type="mse",
                        use_logsoftmax=False, sigma=0.1, beta_kl=1.0, beta_rec=1.0, dropout=0.0, dec_bone="gauss",
                        patch_size=16, topk=15, n_kp_enc=20, eval_epoch_freq=5,
                        learned_feature_dim=0, n_kp_prior=100, weight_decay=0.0, kp_range=(0, 1),
                        run_prefix="", mask_threshold=0.2, use_tps=False, use_pairs=False, use_object_enc=True,
                        use_object_dec=False, warmup_epoch=5, iou_thresh=0.15, anchor_s=0.25, learn_order=False,
                        kl_balance=0.1):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # in conf: "num_processes": 2
    # load data
    if ds == "playground":
        image_size = 64
        enc_channels = (16, 16, 32)
        prior_channels = enc_channels
        # prior_channels = (32, 64)
        ch = 3
        # path_to_data_pickle = './playground_ep_500_steps_20_ra_1_traj_waction_rotate_True_rectangle_ball_tri_s.pickle'
        # path_to_data_pickle = '/mnt/data/tal/box2dplayground/playground_ep_500.pickle'
        path_to_data_pickle = '/media/newhd/data/playground/playground_ep_500.pickle'
        dataset = PlaygroundTrajectoryDataset(path_to_data_pickle, image_size=image_size, timestep_horizon=2,
                                              with_actions=True, traj_length=20)
        milestones = (50, 100, 200)
    elif ds == "celeba":
        image_size = 128
        imwidth = 160
        crop = 16
        ch = 3
        # enc_channels = [64, 128, 256, 512]
        enc_channels = [32, 64, 128, 256]
        # prior_channels = (16, 16, 32)
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/celeba'
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
    elif ds == 'replay_buffer':
        image_size = 64
        ch = 3
        enc_channels = [32, 64, 128]
        prior_channels = (16, 32, 64)
        root = '../../../SAC-AE/pytorch_sac_ae/data/cheetah_run'
        dataset = ReplayBufferDataset(path_to_dir=root, image_size=image_size, obs_shape=(84, 84, 3))
        milestones = (50, 80, 100)
    elif ds == 'bair':
        image_size = 64
        ch = 3
        enc_channels = (32, 64, 128)
        prior_channels = (16, 32, 64)
        root = '/mnt/data/tal/bair/processed'
        dataset = BAIRDataset(root=root, train=True, horizon=2, image_size=image_size)
        milestones = (50, 100, 200)
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
    else:
        raise NotImplementedError

    # TODO: add tensorboard
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
               'prior_channels': prior_channels}

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True,
                            drop_last=True)
    # model
    model = KeyPointVAE(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                        image_size=image_size, n_kp=n_kp, learned_feature_dim=learned_feature_dim,
                        use_logsoftmax=use_logsoftmax, pad_mode=pad_mode, sigma=sigma,
                        dropout=dropout, dec_bone=dec_bone, patch_size=patch_size, n_kp_enc=n_kp_enc,
                        n_kp_prior=n_kp_prior, kp_range=kp_range, kp_activation=kp_activation,
                        mask_threshold=mask_threshold, use_object_enc=use_object_enc,
                        use_object_dec=use_object_dec, anchor_s=anchor_s, learn_order=learn_order)
    logvar_p = torch.log(torch.tensor(sigma ** 2)).to(accelerator.device)  # logvar of the constant std -> for the kl
    # prepare saving location
    run_name = f'{ds}_var_particles_{dec_bone}' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')
    save_config(log_dir, hparams)

    kl_loss_func = ChamferLossKL(use_reverse_kl=False)
    if recon_loss_type == "vgg":
        recon_loss_func = VGGDistance(device=accelerator.device)
    else:
        recon_loss_func = calc_reconstruction_loss
    # original
    # optimizer = optim.Adam(model.get_parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=(100, 200, 400), gamma=0.1)
    # for soft-introvae
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
                torch.load(os.path.join(save_dir, f'{ds}_var_particles_{dec_bone}.pth'),
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
        # debug
        # wait an unwrap model
        # accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        # valid_loss = evaluate_validation_elbo(unwrapped_model, ds, epoch, batch_size=batch_size,
        #                                       recon_loss_type=recon_loss_type, device=accelerator.device,
        #                                       save_image=True, fig_dir=fig_dir, topk=topk,
        #                                       recon_loss_func=recon_loss_func)
        # accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)

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
            # cropped_objects_alpha = model_output['cropped_objects_alpha']
            # cropped_objects_masks = model_output['cropped_objects_masks']
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
            # features_prior = torch.zeros(size=(mu_p.shape[0], mu_p.shape[1], learned_feature_dim),
            #                              device=accelerator.device).float()
            # if learned_feature_dim > 0:
            #     mu_post = torch.cat([mu, mu_features], dim=-1)
            #     logvar_post = torch.cat([logvar, logvar_features], dim=-1)
            #     mu_prior = torch.cat([mu_p, features_prior], dim=-1)
            #     logvar_prior = torch.cat([logvar_kp, logvar_p.expand_as(features_prior)], dim=-1)
            # else:
            #     mu_post = mu
            #     logvar_post = logvar
            #     mu_prior = mu_p
            #     logvar_prior = logvar_kp

            mu_post = mu
            logvar_post = logvar
            mu_prior = mu_p
            logvar_prior = logvar_kp

            loss_kl_kp = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
                                      logvar_gts=logvar_prior).mean()

            # if kl_type == "chamfer":
            #     loss_kl = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
            #                            logvar_gts=logvar_prior).mean()
            # else:
            #     loss_kl = calc_kl(logvar_post.reshape(batch_size, -1), mu_post.reshape(batch_size, -1),
            #                       mu_o=mu_p.reshape(batch_size, -1),
            #                       logvar_o=logvar_prior.reshape(batch_size, -1),
            #                       reduce='mean')
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
            # loss.backward()
            accelerator.backward(loss)
            # if dec_bone in ["gauss", "gauss_feat", "gauss_pointent"]:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        # update bb scores
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
        # accelerator.print(f'epoch {epoch} summary for dec backbone: {dec_bone}')
        # accelerator.print(f'loss: {losses[-1]:.3f}, rec: {losses_rec[-1]:.3f}, kl: {losses_kl[-1]:.3f}')
        # accelerator.print(f'mu max: {mu.max()}, mu min: {mu.min()}')
        # accelerator.print(f'kl_balance: {kl_balance:.3f}, kl_kp: {losses_kl_kp[-1]:.3f},'
        #                   f' kl_feat: {losses_kl_feat[-1]:.3f}')
        # if ds != 'celeba':
        #     accelerator.print(f'val loss (freq: {eval_epoch_freq}): {valid_loss:.3f},'
        #                       f' best: {best_valid_loss:.3f} @ epoch: {best_valid_epoch}')
        # if obj_on is not None:
        #     accelerator.print(f'obj_on max: {obj_on.max()}, obj_on min: {obj_on.min()}')
        # wait an unwrap model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            if accelerator.is_main_process:
                max_imgs = 8
                img_with_kp = plot_keypoints_on_image_batch(mu.clamp(min=kp_range[0], max=kp_range[1]), x, radius=3,
                                                            thickness=1, max_imgs=max_imgs, kp_range=kp_range)
                img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x_prior, radius=3, thickness=1, max_imgs=max_imgs,
                                                              kp_range=kp_range)
                # top-k
                with torch.no_grad():
                    logvar_sum = logvar.sum(-1)
                    logvar_topk = torch.topk(logvar_sum, k=topk, dim=-1, largest=False)
                    indices = logvar_topk[1]  # [batch_size, topk]
                    batch_indices = torch.arange(mu.shape[0]).view(-1, 1).to(mu.device)
                    topk_kp = mu[batch_indices, indices]
                    # bounding boxes
                    masks = create_masks_fast(mu[:, :-1].detach(), anchor_s=unwrapped_model.anchor_s,
                                              feature_dim=x.shape[-1])
                    masks = torch.where(masks < mask_threshold, 0.0, 1.0)
                    bb_scores = -1 * logvar_sum
                # img_with_masks_nms, nms_ind = plot_bb_on_image_batch_from_masks_nms(masks, x, scores=bb_scores,
                #                                                                     iou_thresh=iou_thresh,
                #                                                                     thickness=1, max_imgs=max_imgs,
                #                                                                     hard_thresh=bb_mean_score)
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
                                                 dec_objects[:max_imgs, -3:]],
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
                                 os.path.join(save_dir, f'{ds}_var_particles_{dec_bone}{run_prefix}.pth'))
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
                                                          f'{ds}_var_particles_{dec_bone}{run_prefix}_best.pth'))
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
                                     os.path.join(save_dir, f'{ds}_var_particles_{dec_bone}{run_prefix}_best.pth'))

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
    lr = 2e-4
    batch_size = 64
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 120
    load_model = False
    eval_epoch_freq = 2

    use_logsoftmax = False
    # pad_mode = 'zeros'
    pad_mode = 'reflect'
    sigma = 0.1  # default sigma for the gaussian maps
    dropout = 0.0

    beta_kl = 30.0  # 0.3
    beta_rec = 1.0

    n_kp = 1  # num kp per patch
    n_kp_enc = 10  # total kp to output from the encoder / filter from prior
    # n_kp_enc = 100
    # n_kp_prior = 200  # total kp to filter from prior
    n_kp_prior = 15
    mask_threshold = 0.2  # mask threshold for the features from the encoder
    patch_size = 8  # 8 for playground, 16 for celeb
    learned_feature_dim = 5  # additional features than x,y for each kp

    # kp_range = (0, 1)
    kp_range = (-1, 1)

    weight_decay = 0.0
    # weight_decay = 5e-4

    dec_bone = "gauss_pointnetpp"
    # dec_bone = "gauss_pointnetpp_feat"

    topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
    # topk = 30

    # ds = 'playground'
    # ds = 'celeba'
    # ds = 'replay_buffer'
    # ds = 'bair'
    ds = 'clevrer'
    # ds = 'traffic'

    kl_type = "chamfer"
    # kl_type = "regular"

    # recon_loss_type = "mse"
    recon_loss_type = "vgg"

    # kp_activation = "none"
    # kp_activation = "sigmoid"
    kp_activation = "tanh"

    # run_prefix = "_soft"
    # run_prefix = ""
    # run_prefix = "_masked"
    run_prefix = ""
    use_ema = False
    beta_ema = 0.99

    use_tps = False
    use_pairs = False

    use_object_enc = True  # separate object encoder
    use_object_dec = True  # separate object decoder
    warmup_epoch = 1
    anchor_s = 0.25
    learn_order = False
    kl_balance = 0.001

    model = train_var_particles(ds=ds, batch_size=batch_size, lr=lr,
                                num_epochs=num_epochs, kp_activation=kp_activation,
                                load_model=load_model, n_kp=n_kp, use_logsoftmax=use_logsoftmax, pad_mode=pad_mode,
                                sigma=sigma, beta_kl=beta_kl, beta_rec=beta_rec, dropout=dropout, dec_bone=dec_bone,
                                kp_range=kp_range, learned_feature_dim=learned_feature_dim, weight_decay=weight_decay,
                                recon_loss_type=recon_loss_type, patch_size=patch_size, topk=topk, n_kp_enc=n_kp_enc,
                                eval_epoch_freq=eval_epoch_freq, n_kp_prior=n_kp_prior, run_prefix=run_prefix,
                                mask_threshold=mask_threshold, use_tps=use_tps, use_pairs=use_pairs, anchor_s=anchor_s,
                                use_object_enc=use_object_enc, use_object_dec=use_object_dec,
                                warmup_epoch=warmup_epoch, learn_order=learn_order, kl_balance=kl_balance)

    # for b_kl in [40.0, 80.0, 100.0, 200.0]:
    #     model = train_var_particles(ds=ds, batch_size=batch_size, lr=lr,
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
