"""
Evaluation of the ELBO on the validation set
"""
# imports
import numpy as np
# torch
import torch
import torch.nn.functional as F
from utils.loss_functions import ChamferLossKL, calc_kl, calc_reconstruction_loss, VGGDistance
from torch.utils.data import DataLoader
import torchvision.utils as vutils
# datasets
from datasets.traffic_ds import TrafficDataset
from datasets.clevrer_ds import CLEVRERDataset
# util functions
from utils.util_func import plot_keypoints_on_image_batch


def evaluate_validation_elbo(model, ds, epoch, batch_size=100, recon_loss_type="vgg", device=torch.device('cpu'),
                             save_image=False, fig_dir='./', topk=5, recon_loss_func=None, beta_rec=1.0, beta_kl=1.0,
                             kl_balance=1.0, accelerator=None):
    model.eval()
    kp_range = model.kp_range
    # load data
    if ds == "traffic":
        image_size = 128
        root = '/mnt/data/tal/traffic_dataset/img128np_fs3.npy'
        mode = 'single'
        dataset = TrafficDataset(path_to_npy=root, image_size=image_size, mode=mode, train=False)
    elif ds == 'clevrer':
        image_size = 128
        root = '/mnt/data/tal/clevrer/clevrer_img128np_fs3_valid.npy'
        # root = '/media/newhd/data/clevrer/valid/clevrer_img128np_fs3_valid.npy'
        mode = 'single'
        dataset = CLEVRERDataset(path_to_npy=root, image_size=image_size, mode=mode, train=False)
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, drop_last=False)
    kl_loss_func = ChamferLossKL(use_reverse_kl=False)
    if recon_loss_func is None:
        if recon_loss_type == "vgg":
            recon_loss_func = VGGDistance(device=device)
        else:
            recon_loss_func = calc_reconstruction_loss

    elbos = []
    for batch in dataloader:
        if ds == 'traffic':
            if mode == 'single':
                x = batch.to(device)
                x_prior = x
            else:
                x = batch[0].to(device)
                x_prior = batch[1].to(device)
        elif ds == 'clevrer':
            if mode == 'single':
                x = batch.to(device)
                x_prior = x
            else:
                x = batch[0].to(device)
                x_prior = batch[1].to(device)
        else:
            x = batch
            x_prior = x
        batch_size = x.shape[0]
        # forward pass
        with torch.no_grad():
            model_output = model(x, x_prior=x_prior)
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

        # reconstruction error
        if recon_loss_type == "vgg":
            loss_rec = recon_loss_func(x, rec_x, reduction="mean")
        else:
            loss_rec = calc_reconstruction_loss(x, rec_x, loss_type='mse', reduction='mean')

        # kl-divergence
        logvar_p = torch.log(torch.tensor(model.sigma ** 2)).to(mu.device)  # logvar of the constant std -> for the kl
        logvar_kp = logvar_p.expand_as(mu_p)

        mu_post = mu
        logvar_post = logvar
        mu_prior = mu_p
        logvar_prior = logvar_kp

        loss_kl_kp = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
                                  logvar_gts=logvar_prior).mean()
        if model.learned_feature_dim > 0:
            loss_kl_feat = calc_kl(logvar_features.view(-1, logvar_features.shape[-1]),
                                   mu_features.view(-1, mu_features.shape[-1]), reduce='none')
            loss_kl_feat = loss_kl_feat.view(batch_size, model.n_kp_enc + 1).sum(1).mean()
        else:
            loss_kl_feat = torch.tensor(0.0, device=mu.device)
        loss_kl = loss_kl_kp + kl_balance * loss_kl_feat
        elbo = beta_rec * loss_rec + beta_kl * loss_kl
        elbos.append(elbo.data.cpu().numpy())
    if save_image:
        max_imgs = 8
        img_with_kp = plot_keypoints_on_image_batch(mu.clamp(min=model.kp_range[0], max=model.kp_range[1]), x, radius=3,
                                                    thickness=1, max_imgs=max_imgs, kp_range=model.kp_range)
        img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x_prior, radius=3, thickness=1, max_imgs=max_imgs,
                                                      kp_range=model.kp_range)
        # top-k
        with torch.no_grad():
            logvar_sum = logvar.sum(-1)
            logvar_topk = torch.topk(logvar_sum, k=topk, dim=-1, largest=False)
            indices = logvar_topk[1]  # [batch_size, topk]
            batch_indices = torch.arange(mu.shape[0]).view(-1, 1).to(mu.device)
            topk_kp = mu[batch_indices, indices]
        img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                         radius=3, thickness=1, max_imgs=max_imgs,
                                                         kp_range=kp_range)
        if model.use_object_dec and dec_objects_original is not None:
            dec_objects = model_output['dec_objects']
            if accelerator is not None:
                if accelerator.is_main_process:
                    vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(mu.device),
                                                 rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(mu.device),
                                                 img_with_kp_topk[:max_imgs, -3:].to(mu.device),
                                                 dec_objects[:max_imgs, -3:]],
                                                dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                      nrow=8, pad_value=1)
            else:
                vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(mu.device),
                                             rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(mu.device),
                                             img_with_kp_topk[:max_imgs, -3:].to(mu.device),
                                             dec_objects[:max_imgs, -3:]],
                                            dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
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
            if accelerator is not None:
                if accelerator.is_main_process:
                    vutils.save_image(
                        torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                                  dim=0).data.cpu(), '{}/image_obj_valid_{}.jpg'.format(fig_dir, epoch),
                        nrow=8, pad_value=1)
            else:
                vutils.save_image(
                    torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                              dim=0).data.cpu(), '{}/image_obj_valid_{}.jpg'.format(fig_dir, epoch),
                    nrow=8, pad_value=1)
        else:
            if accelerator is not None:
                if accelerator.is_main_process:
                    vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(mu.device),
                                                 rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(mu.device),
                                                 img_with_kp_topk[:max_imgs, -3:].to(mu.device)],
                                                dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                      nrow=8, pad_value=1)
            else:
                vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(mu.device),
                                             rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(mu.device),
                                             img_with_kp_topk[:max_imgs, -3:].to(mu.device)],
                                            dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                  nrow=8, pad_value=1)
    return np.mean(elbos)
