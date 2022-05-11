"""
CelebA dataset processing
classes and functions from:
https://github.com/jamt9000/DVE
"""

import numpy as np
import pandas as pd
import os
from PIL import Image
import utils.tps as tps
import glob
import torch
from os.path import join as pjoin
from scipy.io import loadmat
import torchvision.transforms as transforms
# from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
# from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sklearn.linear_model import Ridge
import cv2

from io import BytesIO
import sys
from pathlib import Path
import matplotlib

import matplotlib.pyplot as plt  # NOQA


def get_top_k_kp(mu, logvar, mu_features=None, logvar_features=None, k=30):
    """
    returns the top-k indices with LOWEST variance (highest confidence)
    """
    with torch.no_grad():
        logvar_sum = logvar.sum(-1)
        logvar_topk = torch.topk(logvar_sum, k=k, dim=-1, largest=False)
        indices = logvar_topk[1]  # [batch_size, topk]
        indices = indices.sort()[0]
        batch_indices = torch.arange(mu.shape[0]).view(-1, 1).to(mu.device)
        topk_mu = mu[batch_indices, indices]
        topk_logvar = logvar[batch_indices, indices]
        if mu_features is not None:
            topk_mu_features = mu_features[batch_indices, indices]
        else:
            topk_mu_features = None
        if logvar_features is not None:
            topk_logvar_features = logvar_features[batch_indices, indices]
        else:
            topk_logvar_features = None
    return topk_mu, topk_logvar, topk_mu_features, topk_logvar_features


def evaluate_lin_reg_on_mafl(model, root="/mnt/data/tal/celeba", bias=False, use_logvar=False, batch_size=100,
                             kp_range=(0, 1), device=torch.device('cpu'), img_size=128, fig_dir=None, epoch=None,
                             use_features=False, topk=0, normalize=None):
    """
    Evaluating on linear regression of keypoints. KP should be in the image scale and not normalized.
    """
    model.eval()  # put model in evaluation mode
    # imwidth = 136
    # crop = 4
    imwidth = 160
    crop = 16
    dataset = MAFLAligned(root, train=True, pair_warper=None, imwidth=imwidth, crop=crop,
                          do_augmentations=False, use_keypoints=True)
    test_dataset = MAFLAligned(root, train=False, pair_warper=None, imwidth=imwidth, crop=crop,
                               do_augmentations=False, use_keypoints=True)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # extract keypoints from model - train data
    x_train_mu = []
    x_train_logvar = []
    x_train_features_mu = []
    x_train_features_logvar = []
    y_train = []
    for batch in train_dataloader:
        img, gt_kp = batch['data'], batch['meta']['keypts']
        gt_kp_t = gt_kp.clone()
        gt_kp_t[:, :, 0] = gt_kp[:, :, 1]  # change x-y
        gt_kp_t[:, :, 1] = gt_kp[:, :, 0]
        gt_kp_t = gt_kp_t.view(gt_kp_t.shape[0], -1)
        y_train.append(gt_kp_t)
        with torch.no_grad():
            if normalize is not None:
                img = normalize(img)
            mu_kp, logvar_kp, kp_heatmap, mu_features, logvar_features, _, _ = model.encode_all(img.to(device),
                                                                                                return_heatmap=True,
                                                                                                deterministic=True)
        if topk > 0:
            mu_kp, logvar_kp, mu_features, logvar_features = get_top_k_kp(mu_kp, logvar_kp, mu_features,
                                                                          logvar_features, k=topk)
        x_train_mu.append(mu_kp)
        x_train_logvar.append(logvar_kp)
        if use_features and mu_features is not None:
            x_train_features_mu.append(mu_features)
            x_train_features_logvar.append(logvar_features)

    x_train_mu = torch.cat(x_train_mu, dim=0)
    x_train_logvar = torch.cat(x_train_logvar, dim=0)
    if use_features and mu_features is not None:
        x_train_features_mu = torch.cat(x_train_features_mu, dim=0)
        x_train_features_logvar = torch.cat(x_train_features_logvar, dim=0)
    if use_logvar:
        x_train = torch.cat([x_train_mu, x_train_logvar], dim=-1)
    else:
        x_train = x_train_mu
    if use_features and mu_features is not None:
        x_train = torch.cat([x_train, x_train_features_mu, x_train_features_logvar], dim=-1)
    x_train = x_train.view(x_train.shape[0], -1).data.cpu().numpy()
    y_train = torch.cat(y_train, dim=0)
    y_train = y_train.data.cpu().numpy()

    # extract keypoints from model - test data
    x_test_mu = []
    x_test_logvar = []
    x_test_features_mu = []
    x_test_features_logvar = []
    y_test = []
    for batch in test_dataloader:
        img, gt_kp = batch['data'], batch['meta']['keypts']
        gt_kp_t = gt_kp.clone()
        gt_kp_t[:, :, 0] = gt_kp[:, :, 1]  # change x-y
        gt_kp_t[:, :, 1] = gt_kp[:, :, 0]
        y_test.append(gt_kp_t)
        with torch.no_grad():
            if normalize is not None:
                img = normalize(img)
            mu_kp, logvar_kp, kp_heatmap, mu_features, logvar_features, _, _ = model.encode_all(img.to(device),
                                                                                                return_heatmap=True,
                                                                                                deterministic=True)
        if topk > 0:
            mu_kp, logvar_kp, mu_features, logvar_features = get_top_k_kp(mu_kp, logvar_kp, mu_features,
                                                                          logvar_features, k=topk)
        x_test_mu.append(mu_kp)
        x_test_logvar.append(logvar_kp)
        if use_features and mu_features is not None:
            x_test_features_mu.append(mu_features)
            x_test_features_logvar.append(logvar_features)

    x_test_mu = torch.cat(x_test_mu, dim=0)
    x_test_logvar = torch.cat(x_test_logvar, dim=0)
    if use_features and mu_features is not None:
        x_test_features_mu = torch.cat(x_test_features_mu, dim=0)
        x_test_features_logvar = torch.cat(x_test_features_logvar, dim=0)
    if use_logvar:
        x_test = torch.cat([x_test_mu, x_test_logvar], dim=-1)
    else:
        x_test = x_test_mu
    if use_features and mu_features is not None:
        x_test = torch.cat([x_test, x_test_features_mu, x_test_features_logvar], dim=-1)
    x_test = x_test.view(x_test.shape[0], -1).data.cpu().numpy()
    y_test = torch.cat(y_test, dim=0)
    y_test = y_test.data.cpu().numpy()

    # regression
    regr = Ridge(alpha=0.0, fit_intercept=bias)
    _ = regr.fit(x_train, y_train)
    y_predict = regr.predict(x_test)
    y_predict_train = regr.predict(x_train)

    landmarks_gt = y_test.astype(np.float32)
    landmarks_regressed = y_predict.reshape(landmarks_gt.shape)

    landmarks_gt_train = y_train.reshape(y_train.shape[0], landmarks_gt.shape[1], landmarks_gt.shape[2])
    landmarks_regressed_train = y_predict_train.reshape(landmarks_gt_train.shape)

    # normalized error with respect to intra-occular distance
    eyes = landmarks_gt[:, :2, :]
    occular_distances = np.sqrt(
        np.sum((eyes[:, 0, :] - eyes[:, 1, :]) ** 2, axis=-1))
    distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed) ** 2, axis=-1))
    mean_error = np.mean(distances / occular_distances[:, None])

    # normalized train error with respect to intra-occular distance
    eyes_train = landmarks_gt_train[:, :2, :]
    occular_distances_train = np.sqrt(
        np.sum((eyes_train[:, 0, :] - eyes_train[:, 1, :]) ** 2, axis=-1))
    distances_train = np.sqrt(np.sum((landmarks_gt_train - landmarks_regressed_train) ** 2, axis=-1))
    mean_error_train = np.mean(distances_train / occular_distances_train[:, None])

    if fig_dir is not None and epoch is not None:
        # save examples
        indices = np.random.randint(low=0, high=y_predict.shape[0] - 1, size=8)
        samples = [test_dataset[i] for i in indices]
        img = [samples[i]["data"] for i in range(len(samples))]
        img = torch.stack(img, dim=0).to(device)
        # print(img.shape)
        gt_kp = torch.from_numpy(landmarks_gt[indices]) / (img_size - 1)
        pred_kp = torch.from_numpy(landmarks_regressed[indices]) / (img_size - 1)
        max_imgs = 8
        if use_logvar:
            img_name = f'{fig_dir}/eval_image_{epoch}_v.jpg'
        else:
            img_name = f'{fig_dir}/eval_image_{epoch}.jpg'
        img_with_gt_kp = plot_keypoints_on_image_batch(gt_kp, img, radius=3, thickness=1, max_imgs=max_imgs,
                                                       kp_range=kp_range)
        img_with_pred_kp = plot_keypoints_on_image_batch(pred_kp, img, radius=3, thickness=1, max_imgs=max_imgs,
                                                         kp_range=kp_range)
        save_image(torch.cat([img[:max_imgs, -3:], img_with_gt_kp[:max_imgs, -3:].to(device),
                              img_with_pred_kp[:max_imgs, -3:].to(device)], dim=0).data.cpu(), img_name, nrow=8,
                   pad_value=1)

    return mean_error_train, mean_error


def lin_reg(x_train, y_train, x_test, y_test, bias=False):
    # regression
    regr = Ridge(alpha=0.0, fit_intercept=bias)
    _ = regr.fit(x_train, y_train)
    y_predict = regr.predict(x_test)
    y_predict_train = regr.predict(x_train)

    landmarks_gt = y_test.astype(np.float32)
    landmarks_regressed = y_predict.reshape(landmarks_gt.shape)

    landmarks_gt_train = y_train.reshape(y_train.shape[0], landmarks_gt.shape[1], landmarks_gt.shape[2])
    landmarks_regressed_train = y_predict_train.reshape(landmarks_gt_train.shape)

    # normalized error with respect to intra-occular distance
    eyes = landmarks_gt[:, :2, :]
    occular_distances = np.sqrt(
        np.sum((eyes[:, 0, :] - eyes[:, 1, :]) ** 2, axis=-1))
    distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed) ** 2, axis=-1))
    mean_error = np.mean(distances / occular_distances[:, None])

    return mean_error


def evaluate_lin_reg_on_mafl_topk(model, root="/mnt/data/tal/celeba", bias=False, batch_size=100,
                                  device=torch.device('cpu'), topk=10, normalize=None):
    """
    Evaluating on linear regression of keypoints. KP should be in the image scale and not normalized.
    """
    model.eval()  # put model in evaluation mode
    imwidth = 160
    crop = 16
    dataset = MAFLAligned(root, train=True, pair_warper=None, imwidth=imwidth, crop=crop,
                          do_augmentations=False, use_keypoints=True)
    test_dataset = MAFLAligned(root, train=False, pair_warper=None, imwidth=imwidth, crop=crop,
                               do_augmentations=False, use_keypoints=True)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # extract keypoints from model - train data
    x_train_mu = []
    x_train_logvar = []
    x_train_features_mu = []
    x_train_features_logvar = []
    y_train = []
    for batch in train_dataloader:
        img, gt_kp = batch['data'], batch['meta']['keypts']
        gt_kp_t = gt_kp.clone()
        gt_kp_t[:, :, 0] = gt_kp[:, :, 1]  # change x-y
        gt_kp_t[:, :, 1] = gt_kp[:, :, 0]
        gt_kp_t = gt_kp_t.view(gt_kp_t.shape[0], -1)
        y_train.append(gt_kp_t)
        with torch.no_grad():
            if normalize is not None:
                img = normalize(img)
            mu_kp, logvar_kp, kp_heatmap, mu_features, logvar_features, _, _ = model.encode_all(img.to(device),
                                                                                                return_heatmap=True,
                                                                                                deterministic=True)
        x_train_mu.append(mu_kp)
        x_train_logvar.append(logvar_kp)

        if mu_features is not None:
            x_train_features_mu.append(mu_features)
            x_train_features_logvar.append(logvar_features)

    x_train_mu = torch.cat(x_train_mu, dim=0)
    x_train_logvar = torch.cat(x_train_logvar, dim=0)
    if mu_features is not None:
        x_train_features_mu = torch.cat(x_train_features_mu, dim=0)
        x_train_features_logvar = torch.cat(x_train_features_logvar, dim=0)

    # top-k variance based
    confidence = x_train_logvar.sum(-1).mean(0)  # [n_kp]
    _, top_confident_kp = torch.topk(confidence, k=topk, largest=False)
    _, top_uncertainty_kp = torch.topk(confidence, k=topk, largest=True)

    x_train = x_train_mu
    x_train_confident = x_train[:, top_confident_kp]
    x_train_uncertainty = x_train[:, top_uncertainty_kp]

    x_train_logvar = torch.cat([x_train_mu, x_train_logvar], dim=-1)
    x_train_logvar_confident = x_train_logvar[:, top_confident_kp]
    x_train_logvar_uncertainty = x_train_logvar[:, top_uncertainty_kp]

    if mu_features is not None:
        x_train_features = torch.cat([x_train_logvar, x_train_features_mu, x_train_features_logvar], dim=-1)
        x_train_features_confident = x_train_features[:, top_confident_kp]
        x_train_features_uncertainty = x_train_features[:, top_uncertainty_kp]

    x_train = x_train.view(x_train.shape[0], -1).data.cpu().numpy()
    x_train_confident = x_train_confident.view(x_train_confident.shape[0], -1).data.cpu().numpy()
    x_train_uncertainty = x_train_uncertainty.view(x_train_uncertainty.shape[0], -1).data.cpu().numpy()

    x_train_logvar = x_train_logvar.view(x_train_logvar.shape[0], -1).data.cpu().numpy()
    x_train_logvar_confident = x_train_logvar_confident.view(x_train_logvar_confident.shape[0], -1).data.cpu().numpy()
    x_train_logvar_uncertainty = x_train_logvar_uncertainty.view(x_train_logvar_uncertainty.shape[0],
                                                                 -1).data.cpu().numpy()

    if mu_features is not None:
        x_train_features = x_train_features.view(x_train_features.shape[0], -1).data.cpu().numpy()
        x_train_features_confident = x_train_features_confident.view(x_train_features_confident.shape[0],
                                                                     -1).data.cpu().numpy()
        x_train_features_uncertainty = x_train_features_uncertainty.view(x_train_features_uncertainty.shape[0],
                                                                         -1).data.cpu().numpy()

    y_train = torch.cat(y_train, dim=0)
    y_train = y_train.data.cpu().numpy()

    # extract keypoints from model - test data
    x_test_mu = []
    x_test_logvar = []
    x_test_features_mu = []
    x_test_features_logvar = []
    y_test = []
    for batch in test_dataloader:
        img, gt_kp = batch['data'], batch['meta']['keypts']
        gt_kp_t = gt_kp.clone()
        gt_kp_t[:, :, 0] = gt_kp[:, :, 1]  # change x-y
        gt_kp_t[:, :, 1] = gt_kp[:, :, 0]
        y_test.append(gt_kp_t)
        with torch.no_grad():
            if normalize is not None:
                img = normalize(img)
            mu_kp, logvar_kp, kp_heatmap, mu_features, logvar_features, _, _ = model.encode_all(img.to(device),
                                                                                                return_heatmap=True,
                                                                                                deterministic=True)
        x_test_mu.append(mu_kp)
        x_test_logvar.append(logvar_kp)
        if mu_features is not None:
            x_test_features_mu.append(mu_features)
            x_test_features_logvar.append(logvar_features)

    x_test_mu = torch.cat(x_test_mu, dim=0)
    x_test_logvar = torch.cat(x_test_logvar, dim=0)
    if mu_features is not None:
        x_test_features_mu = torch.cat(x_test_features_mu, dim=0)
        x_test_features_logvar = torch.cat(x_test_features_logvar, dim=0)

    x_test = x_test_mu
    x_test_confident = x_test[:, top_confident_kp]
    x_test_uncertainty = x_test[:, top_uncertainty_kp]

    x_test_logvar = torch.cat([x_test_mu, x_test_logvar], dim=-1)
    x_test_logvar_confident = x_test_logvar[:, top_confident_kp]
    x_test_logvar_uncertainty = x_test_logvar[:, top_uncertainty_kp]

    if mu_features is not None:
        x_test_features = torch.cat([x_test_logvar, x_test_features_mu, x_test_features_logvar], dim=-1)
        x_test_features_confident = x_test_features[:, top_confident_kp]
        x_test_features_uncertainty = x_test_features[:, top_uncertainty_kp]

    x_test = x_test.view(x_test.shape[0], -1).data.cpu().numpy()
    x_test_confident = x_test_confident.view(x_test_confident.shape[0], -1).data.cpu().numpy()
    x_test_uncertainty = x_test_uncertainty.view(x_test_uncertainty.shape[0], -1).data.cpu().numpy()

    x_test_logvar = x_test_logvar.view(x_test_logvar.shape[0], -1).data.cpu().numpy()
    x_test_logvar_confident = x_test_logvar_confident.view(x_test_logvar_confident.shape[0], -1).data.cpu().numpy()
    x_test_logvar_uncertainty = x_test_logvar_uncertainty.view(x_test_logvar_uncertainty.shape[0],
                                                               -1).data.cpu().numpy()

    if mu_features is not None:
        x_test_features = x_test_features.view(x_test_features.shape[0], -1).data.cpu().numpy()
        x_test_features_confident = x_test_features_confident.view(x_test_features_confident.shape[0],
                                                                   -1).data.cpu().numpy()
        x_test_features_uncertainty = x_test_features_uncertainty.view(x_test_features_uncertainty.shape[0],
                                                                       -1).data.cpu().numpy()

    y_test = torch.cat(y_test, dim=0)
    y_test = y_test.data.cpu().numpy()

    # regression
    # only mu
    mu_err = lin_reg(x_train, y_train, x_test, y_test, bias=bias)
    mu_confident_err = lin_reg(x_train_confident, y_train, x_test_confident, y_test, bias=bias)
    mu_uncertainty_err = lin_reg(x_train_uncertainty, y_train, x_test_uncertainty, y_test, bias=bias)

    # +logvar
    logvar_err = lin_reg(x_train_logvar, y_train, x_test_logvar, y_test, bias=bias)
    logvar_confident_err = lin_reg(x_train_logvar_confident, y_train, x_test_logvar_confident, y_test, bias=bias)
    logvar_uncertainty_err = lin_reg(x_train_logvar_uncertainty, y_train, x_test_logvar_uncertainty, y_test, bias=bias)

    if mu_features is not None:
        # +features
        feat_err = lin_reg(x_train_features, y_train, x_test_features, y_test, bias=bias)
        feat_confident_err = lin_reg(x_train_features_confident, y_train, x_test_features_confident, y_test, bias=bias)
        feat_uncertainty_err = lin_reg(x_train_features_uncertainty, y_train, x_test_features_uncertainty, y_test,
                                       bias=bias)
    else:
        feat_err = None
        feat_confident_err = None
        feat_uncertainty_err = None

    result = {'mu_err': mu_err, 'mu_confident_err': mu_confident_err, 'mu_uncertainty_err': mu_uncertainty_err,
              'logvar_err': logvar_err, 'logvar_confident_err': logvar_confident_err,
              'logvar_uncertainty_err': logvar_uncertainty_err, 'feat_err': feat_err,
              'feat_confident_err': feat_confident_err, 'feat_uncertainty_err': feat_uncertainty_err}



    return result


# code duplicate from `utils` so it is self-contained.
def color_map(num=100):
    colormap = ["FF355E",
                "8ffe09",
                "1d5dec",
                "FF9933",
                "FFFF66",
                "CCFF00",
                "AAF0D1",
                "FF6EFF",
                "FF00CC",
                "299617",
                "AF6E4D"] * num
    s = ''
    for color in colormap:
        s += color
    b = bytes.fromhex(s)
    cm = np.frombuffer(b, np.uint8)
    cm = cm.reshape(len(colormap), 3)
    return cm


def plot_keypoints_on_image(k, image_tensor, radius=1, thickness=1, kp_range=(0, 1)):
    # https://github.com/DuaneNielsen/keypoints
    height, width = image_tensor.size(1), image_tensor.size(2)
    num_keypoints = k.size(0)

    if len(k.shape) != 2:
        raise Exception('Individual images and keypoints, not batches')

    k = k.clone()
    k[:, 0] = ((k[:, 0] - kp_range[0]) / (kp_range[1] - kp_range[0])) * (height - 1)
    k[:, 1] = ((k[:, 1] - kp_range[0]) / (kp_range[1] - kp_range[0])) * (width - 1)
    # k.floor_()
    k.round_()
    k = k.detach().cpu().numpy()
    #     print(k)

    img = transforms.ToPILImage()(image_tensor.cpu())

    img = np.array(img)
    cmap = color_map()
    cm = cmap[:num_keypoints].astype(int)
    count = 0
    for co_ord, color in zip(k, cm):
        c = color.item(0), color.item(1), color.item(2)
        co_ord = co_ord.squeeze()
        cv2.circle(img, (co_ord[1], co_ord[0]), radius, c, thickness)
        # cv2.circle(img, (co_ord[0], co_ord[1]), radius, c, thickness)
        count += 1

    return img


def plot_keypoints_on_image_batch(kp_batch_tensor, img_batch_tensor, radius=1, thickness=1, max_imgs=8,
                                  kp_range=(0, 1)):
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_kp = []
    for i in range(num_plot):
        img_np = plot_keypoints_on_image(kp_batch_tensor[i], img_batch_tensor[i], radius=radius, thickness=thickness,
                                         kp_range=kp_range)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_kp.append(img_tensor.permute(2, 0, 1))
    img_with_kp = torch.stack(img_with_kp, dim=0)
    return img_with_kp


def pad_and_crop(im, rr):
    """Return im[rr[0]:rr[1],rr[2]:rr[3]]
    Pads if necessary to allow out of bounds indexing
    """

    meanval = np.array(np.dstack((0, 0, 0)), dtype=im.dtype)

    if rr[0] < 0:
        top = -rr[0]
        P = np.tile(meanval, [top, im.shape[1], 1])
        im = np.vstack([P, im])
        rr[0] = rr[0] + top
        rr[1] = rr[1] + top

    if rr[2] < 0:
        left = -rr[2]
        P = np.tile(meanval, [im.shape[0], left, 1])
        im = np.hstack([P, im])
        rr[2] = rr[2] + left
        rr[3] = rr[3] + left

    if rr[1] > im.shape[0]:
        bottom = rr[1] - im.shape[0]
        P = np.tile(meanval, [bottom, im.shape[1], 1])
        im = np.vstack([im, P])

    if rr[3] > im.shape[1]:
        right = rr[3] - im.shape[1]
        P = np.tile(meanval, [im.shape[0], right, 1])
        im = np.hstack([im, P])

    im = im[rr[0]:rr[1], rr[2]:rr[3]]

    return im


def label_colormap(x):
    colors = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
    ])
    ndim = len(x.shape)
    num_classes = 11
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    r = x.clone().float()
    g = x.clone().float()
    b = x.clone().float()
    if ndim == 2:
        rgb = torch.zeros((x.shape[0], x.shape[1], 3))
    else:
        rgb = torch.zeros((x.shape[0], 3, x.shape[2], x.shape[3]))
    colors = torch.from_numpy(colors)
    label_colours = dict(zip(range(num_classes), colors))

    for l in range(0, num_classes):
        r[x == l] = label_colours[l][0]
        g[x == l] = label_colours[l][1]
        b[x == l] = label_colours[l][2]
    if ndim == 2:
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
    elif ndim == 4:
        rgb[:, 0, None] = r / 255.0
        rgb[:, 1, None] = g / 255.0
        rgb[:, 2, None] = b / 255.0
    else:
        import ipdb;
        ipdb.set_trace()
    return rgb


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)


def norm_range(t, range=None):
    t = t.clone()
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))
    return t


class PcaAug(object):
    _eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    _eigvec = torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, im):
        alpha = torch.randn(3) * self.alpha
        rgb = (self._eigvec * alpha.expand(3, 3) * self._eigval.expand(3, 3)).sum(1)
        return im + rgb.reshape(3, 1, 1)


class JPEGNoise(object):
    def __init__(self, low=30, high=99):
        self.low = low
        self.high = high

    def __call__(self, im):
        H = im.height
        W = im.width
        rW = max(int(0.8 * W), int(W * (1 + 0.5 * torch.randn([]))))
        im = TF.resize(im, (rW, rW))
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=torch.randint(self.low, self.high,
                                                          []).item())
        im = Image.open(buf)
        im = TF.resize(im, (H, W))
        return im


def kp_normalize(H, W, kp):
    kp = kp.clone()
    # kp[..., 0] = 2. * kp[..., 0] / (W - 1) - 1
    # kp[..., 1] = 2. * kp[..., 1] / (H - 1) - 1
    kp[..., 0] = kp[..., 0] / (W - 1)
    kp[..., 1] = kp[..., 1] / (H - 1)
    return kp


class CelebABase(Dataset):

    def __len__(self):
        return len(self.filenames)

    def restrict_annos(self, num):
        anno_count = len(self.filenames)
        pick = np.random.choice(anno_count, num, replace=False)
        print(f"Picking annotation for images: {np.array(self.filenames)[pick].tolist()}")
        # exit(0)
        repeat = int(anno_count // num)
        self.filenames = np.tile(np.array(self.filenames)[pick], repeat)
        self.keypoints = np.tile(self.keypoints[pick], (repeat, 1, 1))

    def __getitem__(self, index):
        if (not self.use_ims and not self.use_keypoints):
            # early exit when caching is used
            return {"data": torch.zeros(3, 1, 1), "meta": {"index": index}}

        if self.use_ims:
            im = Image.open(os.path.join(self.subdir, self.filenames[index]))
        # print("imread: {:.3f}s".format(time.time() - tic)) ; tic = time.time()
        kp = None
        if self.use_keypoints:
            kp = self.keypoints[index].copy()
        meta = {}

        if self.warper is not None:
            if self.warper.returns_pairs:
                # tic = time.time()
                im1 = self.initial_transforms(im.convert("RGB"))
                # print("tx1: {:.3f}s".format(time.time() - tic)) ; tic = time.time()
                im1 = TF.to_tensor(im1) * 255
                if False:
                    plt.imshow(norm_range(im1).permute(1, 2, 0).cpu().numpy())
                    plt.scatter(kp[:, 0], kp[:, 1])

                im1, im2, flow, grid, kp1, kp2 = self.warper(im1, keypts=kp, crop=self.crop)
                # print("warper: {:.3f}s".format(time.time() - tic)) ; tic = time.time()

                im1 = im1.to(torch.uint8)
                im2 = im2.to(torch.uint8)

                C, H, W = im1.shape

                im1 = TF.to_pil_image(im1)
                im2 = TF.to_pil_image(im2)

                im1 = self.transforms(im1)
                im2 = self.transforms(im2)
                # print("tx-2: {:.3f}s".format(time.time() - tic)) ; tic = time.time()

                C, H, W = im1.shape
                data = torch.stack((im1, im2), 0)
                meta = {
                    'flow': flow[0],
                    'grid': grid[0],
                    'im1': im1,
                    'im2': im2,
                    'index': index
                }
                if self.use_keypoints:
                    meta = {**meta, **{'kp1': kp1, 'kp2': kp2}}
            else:
                im1 = self.initial_transforms(im.convert("RGB"))
                im1 = TF.to_tensor(im1) * 255

                im1, kp = self.warper(im1, keypts=kp, crop=self.crop)

                im1 = im1.to(torch.uint8)
                im1 = TF.to_pil_image(im1)
                im1 = self.transforms(im1)

                C, H, W = im1.shape
                data = im1
                if self.use_keypoints:
                    meta = {
                        'keypts': kp,
                        'keypts_normalized': kp_normalize(H, W, kp),
                        'index': index
                    }

        else:
            if self.use_ims:
                data = self.transforms(self.initial_transforms(im.convert("RGB")))
                if self.crop != 0:
                    data = data[:, self.crop:-self.crop, self.crop:-self.crop]
                C, H, W = data.shape
            else:
                #  after caching descriptors, there is no point doing I/O
                H = W = self.imwidth - 2 * self.crop
                data = torch.zeros(3, 1, 1)

            if kp is not None:
                kp = kp - self.crop
                kp = torch.tensor(kp)

            if self.use_keypoints:
                meta = {
                    'keypts': kp,
                    'keypts_normalized': kp_normalize(H, W, kp),
                    'index': index
                }
        if self.visualize:
            num_show = 2 if self.warper and self.warper.returns_pairs else 1
            plt.clf()
            fig = plt.figure()
            for ii in range(num_show):
                im_ = data[ii] if num_show > 1 else data
                ax = fig.add_subplot(1, num_show, ii + 1)
                ax.imshow(norm_range(im_).permute(1, 2, 0).cpu().numpy())
                if self.use_keypoints:
                    if num_show == 2:
                        kp_x = meta["kp{}".format(ii + 1)][:, 0].numpy()
                        kp_y = meta["kp{}".format(ii + 1)][:, 1].numpy()
                    else:
                        kp_x = kp[:, 0].numpy()
                        kp_y = kp[:, 1].numpy()
                    ax.scatter(kp_x, kp_y)
        return {"data": data, "meta": meta}


class ProfileData(Dataset):
    def __init__(self, imwidth, **kwargs):
        self.imwidth = imwidth

    def __getitem__(self, index):
        data = torch.randn(3, self.imwidth, self.imwidth)
        return {"data": data}

    def __len__(self):
        return int(1E6)


class AFLW(CelebABase):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, imwidth, train, pair_warper, visualize=False, use_ims=True,
                 use_keypoints=False, do_augmentations=False, crop=0, use_minival=False,
                 **kwargs):
        self.root = root
        self.crop = crop
        self.imwidth = imwidth
        self.use_ims = use_ims
        self.visualize = visualize
        self.use_keypoints = use_keypoints
        self.use_minival = use_minival
        self.train = train
        self.warper = pair_warper

        images, keypoints, sizes = self.load_dataset(root)
        self.sizes = sizes
        self.filenames = images
        self.keypoints = keypoints.astype(np.float32)
        self.subdir = os.path.join(root, 'output')

        # print("LIMITING DATA FOR DEBGGING")
        # self.filenames = self.filenames[:1000]
        # self.keypoints = self.keypoints[:1000]
        # sizes = sizes[:1000]
        # self.sizes = sizes

        # check raw
        # im_path = pjoin(self.subdir, self.filenames[0])
        # im = Image.open(im_path).convert("RGB")
        # plt.imshow(im)
        # plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1])
        self.keypoints *= self.imwidth / sizes[:, [1, 0]].reshape(-1, 1, 2)

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        # NOTE: we break the aspect ratio here, but hopefully the network should
        # be fairly tolerant to this
        self.initial_transforms = transforms.Resize((self.imwidth, self.imwidth))
        augmentations = [
            JPEGNoise(),
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]
        self.transforms = transforms.Compose(augmentations + [normalize])

    def load_dataset(self, data_dir):
        # borrowed from Tom and Ankush
        if self.train or self.use_minival:
            load_subset = "train"
        else:
            load_subset = "test"
        with open(pjoin(data_dir, 'aflw_{}_images.txt'.format(load_subset)), 'r') as f:
            images = f.read().splitlines()
        mat = loadmat(os.path.join(data_dir, 'aflw_' + load_subset + '_keypoints.mat'))
        keypoints = mat['gt'][:, :, [1, 0]]
        sizes = mat['hw']

        # import ipdb; ipdb.set_trace()
        # if self.data.shape[0] == 19000:
        #     self.data = self.data[:20]

        if load_subset == 'train':
            # put the last 10 percent of the training aside for validation
            if self.use_minival:
                n_validation = int(round(0.1 * len(images)))
                if self.train:
                    images = images[:-n_validation]
                    keypoints = keypoints[:-n_validation]
                    sizes = sizes[:-n_validation]
                else:
                    images = images[-n_validation:]
                    keypoints = keypoints[-n_validation:]
                    sizes = sizes[-n_validation:]
        return images, keypoints, sizes


class CelebAPrunedAligned_MAFLVal(CelebABase):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_warper=None, imwidth=100, crop=18,
                 do_augmentations=True, use_keypoints=False, use_hq_ims=True,
                 visualize=False, use_ims=True, val_split="celeba", val_size=2000,
                 **kwargs):
        self.root = root
        self.imwidth = imwidth
        self.train = train
        self.use_ims = use_ims
        self.warper = pair_warper
        self.visualize = visualize
        self.crop = crop
        self.use_keypoints = use_keypoints

        if use_hq_ims:
            subdir = "img_align_celeba_hq"
        else:
            subdir = "img_align_celeba"
        self.subdir = os.path.join(root, 'Img', subdir)

        anno = pd.read_csv(
            os.path.join(root, 'Anno', 'list_landmarks_align_celeba.txt'), header=1,
            delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(root, 'Eval', 'list_eval_partition.txt'),
                            header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltrain = pd.read_csv(os.path.join(root, 'MAFL', 'training.txt'), header=None,
                                delim_whitespace=True, index_col=0)
        mafltest = pd.read_csv(os.path.join(root, 'MAFL', 'testing.txt'), header=None,
                               delim_whitespace=True, index_col=0)
        # Ensure that we are not using mafl images
        split.loc[mafltrain.index] = 3
        split.loc[mafltest.index] = 4

        assert (split[1] == 4).sum() == 1000

        if train:
            self.data = anno.loc[split[split[1] == 0].index]
        elif val_split == "celeba":
            # subsample images from CelebA val, otherwise training gets slow
            self.data = anno.loc[split[split[1] == 2].index][:val_size]
        elif val_split == "mafl":
            self.data = anno.loc[split[split[1] == 4].index]

        # lefteye_x lefteye_y ; righteye_x righteye_y ; nose_x nose_y ;
        # leftmouth_x leftmouth_y ; rightmouth_x rightmouth_y
        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1, 5, 2)
        self.filenames = list(self.data.index)

        # Move head up a bit
        initial_crop = lambda im: transforms.functional.crop(im, 30, 0, 178, 178)
        self.keypoints[:, :, 1] -= 30
        self.keypoints *= self.imwidth / 178.

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        augmentations = [
            JPEGNoise(),
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose(
            [initial_crop, transforms.Resize(self.imwidth)])
        # self.transforms = transforms.Compose(augmentations + [normalize])
        self.transforms = transforms.Compose(augmentations)

    def __len__(self):
        return len(self.data.index)


class MAFLAligned(CelebABase):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_warper=None, imwidth=100, crop=18,
                 do_augmentations=True, use_keypoints=False, use_hq_ims=True,
                 use_ims=True, visualize=False, **kwargs):
        self.root = root
        self.imwidth = imwidth
        self.use_hq_ims = use_hq_ims
        self.use_ims = use_ims
        self.visualize = visualize
        self.train = train
        self.warper = pair_warper
        self.crop = crop
        self.use_keypoints = use_keypoints
        subdir = "img_align_celeba_hq" if use_hq_ims else "img_align_celeba"
        self.subdir = os.path.join(root, 'Img', subdir)
        annos_path = os.path.join(root, 'Anno', 'list_landmarks_align_celeba.txt')
        anno = pd.read_csv(annos_path, header=1, delim_whitespace=True)

        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(root, 'Eval', 'list_eval_partition.txt'),
                            header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599
        mafltest = pd.read_csv(os.path.join(root, 'MAFL', 'testing.txt'), header=None,
                               delim_whitespace=True, index_col=0)
        split.loc[mafltest.index] = 4
        mafltrain = pd.read_csv(os.path.join(root, 'MAFL', 'training.txt'), header=None,
                                delim_whitespace=True, index_col=0)
        split.loc[mafltrain.index] = 5
        assert (split[1] == 4).sum() == 1000
        assert (split[1] == 5).sum() == 19000

        if train:
            self.data = anno.loc[split[split[1] == 5].index]
        else:
            self.data = anno.loc[split[split[1] == 4].index]

        # keypoint ordering
        # lefteye_x lefteye_y ; righteye_x righteye_y ; nose_x nose_y ;
        # leftmouth_x leftmouth_y ; rightmouth_x rightmouth_y
        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1, 5, 2)
        self.filenames = list(self.data.index)

        # Move head up a bit
        vertical_shift = 30
        # crop_params = dict(i=vertical_shift, j=0, h=178, w=178)
        initial_crop = lambda im: transforms.functional.crop(im, 30, 0, 178, 178)
        # initial_crop = lambda im: transforms.functional.crop(im, **crop_params)
        self.keypoints[:, :, 1] -= vertical_shift
        self.keypoints *= self.imwidth / 178.
        # normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
        #                                  std=[0.2599, 0.2371, 0.2323])
        augmentations = [
            JPEGNoise(),
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose(
            [initial_crop, transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose(augmentations)


class AFLW_MTFL(CelebABase):
    """Used for testing on the 5-point version of AFLW included in the MTFL download from the
       Facial Landmark Detection by Deep Multi-task Learning (TCDCN) paper
       http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
       For training this uses a cropped 5-point version of AFLW used in
       http://openaccess.thecvf.com/content_ICCV_2017/papers/Thewlis_Unsupervised_Learning_of_ICCV_2017_paper.pdf
       """
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_warper=None, imwidth=70, use_ims=True,
                 crop=0, do_augmentations=True, use_keypoints=False, visualize=False, **kwargs):
        # MTFL from http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip
        self.test_root = os.path.join(root, 'MTFL')
        # AFLW cropped from www.robots.ox.ac.uk/~jdt/aflw_10122train_cropped.zip
        self.train_root = os.path.join(root, 'aflw_cropped')

        self.imwidth = imwidth
        self.use_ims = use_ims
        self.train = train
        self.warper = pair_warper
        self.crop = crop
        self.use_keypoints = use_keypoints
        self.visualize = visualize
        initial_crop = lambda im: im

        test_anno = pd.read_csv(os.path.join(self.test_root, 'testing.txt'),
                                header=None, delim_whitespace=True)

        if train:
            self.root = self.train_root
            all_anno = pd.read_csv(os.path.join(self.train_root, 'facedata_cropped.csv'),
                                   sep=',', header=0)
            allims = all_anno.image_file.to_list()
            trainims = all_anno[all_anno.set == 1].image_file.to_list()
            testims = [t.split('-')[-1] for t in test_anno.loc[:, 0].to_list()]

            for x in trainims:
                assert x not in testims

            for x in testims:
                assert x in allims

            self.filenames = all_anno[all_anno.set == 1].crop_file.to_list()
            self.keypoints = np.array(all_anno[all_anno.set == 1].iloc[:, 4:14],
                                      dtype=np.float32).reshape(-1, 5, 2)

            self.keypoints -= 1  # matlab to python
            self.keypoints *= self.imwidth / 150.

            assert len(self.filenames) == 10122
        else:
            self.root = self.test_root
            keypoints = np.array(test_anno.iloc[:, 1:11], dtype=np.float32)
            self.keypoints = keypoints.reshape(-1, 2, 5).transpose(0, 2, 1)
            self.filenames = test_anno[0].to_list()

            self.keypoints -= 1  # matlab to python
            self.keypoints *= self.imwidth / 150.

            assert len(self.filenames) == 2995
        self.subdir = self.root

        # print("HARDCODING DEBGGER")
        # self.filenames = self.filenames[:100]
        # self.keypoints = self.keypoints[:100]

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        augmentations = [
            JPEGNoise(),
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]
        self.initial_transforms = transforms.Compose(
            [initial_crop, transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose(augmentations + [normalize])


if __name__ == '__main__':
    pass
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", default="CelebAPrunedAligned_MAFLVal")
    # parser.add_argument("--train", action="store_true")
    # parser.add_argument("--use_keypoints", action="store_true")
    # parser.add_argument("--use_ims", type=int, default=1)
    # parser.add_argument("--use_minival", action="store_true")
    # parser.add_argument("--break_preproc", action="store_true")
    # parser.add_argument("--pairs", action="store_true")
    # parser.add_argument("--rand_in", action="store_true")
    # parser.add_argument("--restrict_to", type=int, help="restrict to n images")
    # parser.add_argument("--downsample_labels", type=int, default=2)
    # parser.add_argument("--show", type=int, default=2)
    # parser.add_argument("--restrict_seed", type=int, default=0)
    # parser.add_argument("--root")
    # args = parser.parse_args()
    #
    # default_roots = {
    #     "CelebAPrunedAligned_MAFLVal": "/mnt/data/tal/celeba",
    #     "MAFLAligned": "/mnt/data/tal/celeba",
    #     "AFLW_MTFL": "data/aflw-mtfl",
    #     "Helen": "data/SmithCVPR2013_dataset_resized",
    #     "AFLW": "data/aflw/aflw_release-2",
    #     "ThreeHundredW": "data/300w/300w",
    #     "Chimps": "data/chimpanzee_faces/datasets_cropped_chimpanzee_faces/data_CZoo",
    # }
    # root = default_roots[args.dataset] if args.root is None else args.root
    #
    # # imwidth = 136
    # imwidth = 160
    # kwargs = {
    #     "root": root,
    #     "train": args.train,
    #     "use_keypoints": args.use_keypoints,
    #     "use_ims": args.use_ims,
    #     "visualize": True,
    #     "use_minival": args.use_minival,
    #     "downsample_labels": args.downsample_labels,
    #     "break_preproc": args.break_preproc,
    #     "rand_in": args.rand_in,
    #     "restrict_to": args.restrict_to,
    #     "restrict_seed": args.restrict_seed,
    #     "imwidth": imwidth,
    #     "crop": 16,  # 20
    # }
    # if args.train and args.pairs:
    #     warper = tps.Warper(H=imwidth, W=imwidth)
    # elif args.train:
    #     # warper = tps.WarperSingle(H=imwidth, W=imwidth)
    #     warper = None
    # else:
    #     print('no warper')
    #     warper = None
    # kwargs["pair_warper"] = warper
    #
    # show = args.show
    # if args.restrict_to:
    #     show = min(args.restrict_to, show)
    # dataset = globals()[args.dataset](**kwargs)
    # print(f'dataset size: {len(dataset)}')
    # for ii in range(show):
    #     print(
    #         f'data shape:{dataset[ii]["data"].shape}, max: {dataset[ii]["data"].max()}, min: {dataset[ii]["data"].min()}')
    #     save_image(dataset[ii]["data"], f'./figures/data_sample_celeb_{ii}.jpg')
    #     if dataset[ii]["meta"]:
    #         # print(f'data shape:{dataset[ii]["meta"].shape}')
    #         print(f'data shape:{dataset[ii]["meta"]}')
    #         if dataset[ii]["meta"].get('keypts_normalized'):
    #             kp = dataset[ii]["meta"]['keypts_normalized'].unsqueeze(0)
    #             kp_t = kp.clone()
    #             kp_t[:, :, 0] = kp[:, :, 1]
    #             kp_t[:, :, 1] = kp[:, :, 0]
    #             img_with_kp = plot_keypoints_on_image_batch(kp_t, dataset[ii]["data"].unsqueeze(0), radius=3,
    #                                                         thickness=1,
    #                                                         max_imgs=1)
    #             save_image(img_with_kp, f'./figures/data_sample_celeb_{ii}_kp.jpg')

    # print(dataset[ii]['data'])
