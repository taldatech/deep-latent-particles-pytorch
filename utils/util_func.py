# imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import datetime
import os
import json
import imageio
# torch
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.ops as ops


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
        count += 1
    #     print(f'{count} kp plotted')

    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # img = np.transpose(img, (2, 0, 1))
    # img = Image.fromarray(img)

    return img


def plot_keypoints_on_image_batch(kp_batch_tensor, img_batch_tensor, radius=1, thickness=1, max_imgs=8,
                                  kp_range=(0, 1)):
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_kp = []
    for i in range(num_plot):
        img_np = plot_keypoints_on_image(kp_batch_tensor[i], img_batch_tensor[i], radius=radius, thickness=1,
                                         kp_range=kp_range)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_kp.append(img_tensor.permute(2, 0, 1))
    img_with_kp = torch.stack(img_with_kp, dim=0)
    return img_with_kp


def plot_batch_kp(img_batch_tensor, kp_batch_tensor, rec_batch_tensor, max_imgs=8):
    batch_size, _, _, im_size = img_batch_tensor.shape
    max_index = (im_size - 1)
    #     print(max_index)
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_kp_np = []
    for i in range(num_plot):
        img_with_kp_np.append(plot_keypoints_on_image(kp_batch_tensor[i], img_batch_tensor[i], radius=2, thickness=1))
    img_np = img_batch_tensor.permute(0, 2, 3, 1).clamp(0, 1).data.cpu().numpy()
    rec_np = rec_batch_tensor.permute(0, 2, 3, 1).clamp(0, 1).data.cpu().numpy()
    fig = plt.figure()
    for i in range(num_plot):
        # image
        ax = fig.add_subplot(3, num_plot, i + 1)
        ax.imshow(img_np[i])
        ax.axis('equal')
        ax.set_axis_off()
        # kp
        ax = fig.add_subplot(3, num_plot, i + 1 + num_plot)
        ax.imshow(img_with_kp_np[i])
        ax.axis('equal')
        ax.set_axis_off()
        # rec
        ax = fig.add_subplot(3, num_plot, i + 1 + 2 * num_plot)
        ax.imshow(rec_np[i])
        ax.axis('equal')
        ax.set_axis_off()
    return fig


def get_kp_mask_from_gmap(gmaps, threshold=0.2, binary=True, elementwise=False):
    """
    gmaps: [B, K, H, W]
    threshold: above it pixels are one and below zero
    """
    if elementwise:
        mask = gmaps
    else:
        mask = gmaps.sum(1, keepdim=True)
    if binary:
        mask = torch.where(mask > threshold, 1.0, 0.0)
    else:
        mask = mask.clamp(0, 1)
        # mask = torch.where(mask > threshold, torch.ones_like(mask), mask)
    return mask


def get_discrete_map(features_map, deterministic=True):
    # features_map: [batch_size, n_kp, features_dim, features_dim]
    batch_size, n_kp, features_dim, _ = features_map.shape
    logits = features_map.permute(0, 2, 3, 1).contiguous()  # [batch_size, features_dim, features_dim, n_kp]
    logits = logits.view(-1, n_kp)  # [batch_size * features_dim * features_dim, n_kp]
    probs = torch.softmax(logits, dim=-1)
    if deterministic:
        sample = torch.argmax(logits, dim=-1)
    else:
        sample = torch.multinomial(probs, num_samples=1)
    sample = F.one_hot(sample, n_kp).squeeze().float().to(features_map.device) + probs - probs.detach()
    sample = sample.view(batch_size, features_dim, features_dim, n_kp).permute(0, 3, 1, 2)
    probs = probs.view(batch_size, features_dim, features_dim, n_kp).permute(0, 3, 1, 2)
    return probs, sample


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu).to(device)
    return mu + eps * std


def create_masks(center, anchor_h, anchor_w, feature_dim=16, kp_range=(-1, 1)):
    # center: [batch_size, n_kp, 2] in kp_range
    # anchor_h, anchor_w: size of anchor in [0, 1]
    batch_size, n_kp = center.shape[0], center.shape[1]
    # transform to (0, 1) range
    center_zo = (center - kp_range[0]) / (kp_range[1] - kp_range[0])
    center_zo = center_zo.view(batch_size * n_kp, -1)
    mask = torch.zeros(batch_size * n_kp, 1, feature_dim, feature_dim, device=center.device).float()
    center_idx = torch.round(center_zo * (feature_dim - 1)).int()
    anchor_hp = np.round(anchor_h * (feature_dim - 1)).astype(int)
    anchor_wp = np.round(anchor_w * (feature_dim - 1)).astype(int)
    #     print(f'center: {center_idx[0]}, anchor h: {anchor_hp}, anchor w: {anchor_wp}')
    ws = (center_idx[:, 0] - anchor_wp // 2).clamp(0, feature_dim).int()
    wt = (center_idx[:, 0] + anchor_wp // 2 + 1).clamp(0, feature_dim).int()
    hs = (center_idx[:, 1] - anchor_hp // 2).clamp(0, feature_dim).int()
    ht = (center_idx[:, 1] + anchor_hp // 2 + 1).clamp(0, feature_dim).int()
    n_bb = ws.shape[0]
    for i in range(n_bb):
        mask[i, 0, ws[i]:wt[i], hs[i]:ht[i]] = 1.0
        # mark center, remove later
        # mask[i, 0, center_idx[i, 1], center_idx[i, 0]] = 0.0
    return mask.view(batch_size, n_kp, 1, feature_dim, feature_dim)


def create_masks_fast(center, anchor_s, feature_dim=16, patch_size=None):
    # center: [batch_size, n_kp, 2] in kp_range
    # anchor_h, anchor_w: size of anchor in [0, 1]
    batch_size, n_kp = center.shape[0], center.shape[1]
    if patch_size is None:
        patch_size = np.round(anchor_s * (feature_dim - 1)).astype(int)
    # create white rectangles
    masks = torch.ones(batch_size * n_kp, 1, patch_size, patch_size, device=center.device).float()
    # pad the masks to image size
    pad_size = (feature_dim - patch_size) // 2
    padded_patches_batch = F.pad(masks, pad=[pad_size] * 4)
    # move the masks to be centered around the kp
    delta_t_batch = 0.0 - center
    delta_t_batch = delta_t_batch.reshape(-1, delta_t_batch.shape[-1])  # [bs * n_kp, 2]
    zeros = torch.zeros([delta_t_batch.shape[0], 1], device=delta_t_batch.device).float()
    ones = torch.ones([delta_t_batch.shape[0], 1], device=delta_t_batch.device).float()
    theta = torch.cat([ones, zeros, delta_t_batch[:, 1].unsqueeze(-1),
                       zeros, ones, delta_t_batch[:, 0].unsqueeze(-1)], dim=-1)
    theta = theta.view(-1, 2, 3)  # [batch_size * n_kp, 2, 3]
    align_corners = False
    padding_mode = 'zeros'
    mode = "nearest"
    # mode = 'bilinear'
    grid = F.affine_grid(theta, padded_patches_batch.size(), align_corners=align_corners)
    trans_padded_patches_batch = F.grid_sample(padded_patches_batch, grid, align_corners=align_corners,
                                               mode=mode, padding_mode=padding_mode)
    trans_padded_patches_batch = trans_padded_patches_batch.view(batch_size, n_kp, *padded_patches_batch.shape[1:])
    # [bs, n_kp, 1, feature_dim, feature_dim]
    return trans_padded_patches_batch


def get_bb_from_masks(masks, width, height):
    # masks: [n_masks, 1, feature_dim, feature_dim]
    n_masks = masks.shape[0]
    mask_h, mask_w = masks.shape[2], masks.shape[3]
    coor = torch.zeros(size=(n_masks, 4), dtype=torch.int, device=masks.device)
    for i in range(n_masks):
        mask = masks[i].int().squeeze()  # [feature_dim, feature_dim]
        indices = (mask == 1).nonzero(as_tuple=False)
        if indices.shape[0] > 0:
            ws = (indices[0][1] * (width / mask_w)).clamp(0, width).int()
            wt = (indices[-1][1] * (width / mask_w)).clamp(0, width).int()
            hs = (indices[0][0] * (height / mask_h)).clamp(0, height).int()
            ht = (indices[-1][0] * (height / mask_h)).clamp(0, height).int()
            coor[i, 0] = ws
            coor[i, 1] = hs
            coor[i, 2] = wt
            coor[i, 3] = ht
    return coor


def get_bb_from_masks_batch(masks, width, height):
    # masks: [batch_size, n_masks, 1, feature_dim, feature_dim]
    coor = torch.zeros(size=(masks.shape[0], masks.shape[1], 4), dtype=torch.int, device=masks.device)
    for i in range(masks.shape[0]):
        coor[i, :, :] = get_bb_from_masks(masks[i], width, height)
    return coor


def nms_single(boxes, scores, iou_thresh=0.5, return_scores=False, remove_ind=None):
    # boxes: [n_bb, 4], scores: [n_boxes]
    nms_indices = ops.nms(boxes.float(), scores, iou_thresh)
    # remove low scoring indices from nms output
    if remove_ind is not None:
        # final_indices = [ind for ind in nms_indices if ind not in remove_ind]
        final_indices = list(set(nms_indices.data.cpu().numpy()) - set(remove_ind))
        # print(f'removed indices: {remove_ind}')
    else:
        final_indices = nms_indices
    nms_boxes = boxes[final_indices]  # [n_bb_nms, 4]
    if return_scores:
        return nms_boxes, final_indices, scores[final_indices]
    else:
        return nms_boxes, final_indices


def remove_low_score_bb_single(boxes, scores, return_scores=False, mode='mean', thresh=0.4, hard_thresh=None):
    # boxes: [n_bb, 4], scores: [n_boxes]
    if hard_thresh is None:
        if mode == 'mean':
            mean_score = scores.mean()
            # indices = (scores > mean_score)
            indices = torch.nonzero(scores > thresh, as_tuple=True)[0].data.cpu().numpy()
        else:
            normalzied_scores = (scores - scores.min()) / (scores.max() - scores.min())
            # indices = (normalzied_scores > thresh)
            indices = torch.nonzero(normalzied_scores > thresh, as_tuple=True)[0].data.cpu().numpy()
    else:
        # indices = (scores > hard_thresh)
        indices = torch.nonzero(scores > hard_thresh, as_tuple=True)[0].data.cpu().numpy()
    # indices = list(range(len(scores)))[indices]
    boxes_t = boxes[indices]
    scores_t = scores[indices]
    if return_scores:
        return indices, boxes_t, scores_t
    else:
        return indices, boxes_t


def get_low_score_bb_single(scores, mode='mean', thresh=0.4, hard_thresh=None):
    # boxes: [n_bb, 4], scores: [n_boxes]
    if hard_thresh is None:
        if mode == 'mean':
            mean_score = scores.mean()
            # indices = (scores > mean_score)
            indices = torch.nonzero(scores < thresh, as_tuple=True)[0].data.cpu().numpy()
        else:
            normalzied_scores = (scores - scores.min()) / (scores.max() - scores.min())
            # indices = (normalzied_scores > thresh)
            indices = torch.nonzero(normalzied_scores < thresh, as_tuple=True)[0].data.cpu().numpy()
    else:
        # indices = (scores > hard_thresh)
        indices = torch.nonzero(scores < hard_thresh, as_tuple=True)[0].data.cpu().numpy()
    # indices = list(range(len(scores)))[indices]
    return indices


def plot_bb_on_image_from_masks_nms(masks, image_tensor, scores, iou_thresh=0.5, thickness=1, hard_thresh=None):
    # masks: [n_masks, 1, feature_dim, feature_dim]
    n_masks = masks.shape[0]
    mask_h, mask_w = masks.shape[2], masks.shape[3]
    height, width = image_tensor.size(1), image_tensor.size(2)
    img = transforms.ToPILImage()(image_tensor.cpu())
    img = np.array(img)
    cmap = color_map()
    cm = cmap[:n_masks].astype(int)
    count = 0
    # get bb coor
    coors = get_bb_from_masks(masks, width, height)  # [n_masks, 4]
    # remove low-score bb
    # remove_ind, coors_top, scores_top = remove_low_score_bb_single(coors, scores, return_scores=True,
    #                                                                mode="mean",
    #                                                                hard_thresh=hard_thresh)  # [n_masks_top, 4]
    # coors_top, scores_top = remove_low_score_bb_single(coors, scores, return_scores=True, mode='minmax')  # [n_masks_top, 4]
    low_score_ind = get_low_score_bb_single(scores, mode='mean', hard_thresh=hard_thresh)
    # nms
    coors_nms, nms_indices, scores_nms = nms_single(coors, scores, iou_thresh, return_scores=True,
                                                    remove_ind=low_score_ind)
    # [n_masks_nms, 4]
    for coor, color in zip(coors_nms, cm):
        c = color.item(0), color.item(1), color.item(2)
        ws = (coor[0] - thickness).clamp(0, width)
        hs = (coor[1] - thickness).clamp(0, height)
        wt = (coor[2] + thickness).clamp(0, width)
        ht = (coor[3] + thickness).clamp(0, height)
        bb_s = (ws, hs)
        bb_t = (wt, ht)
        cv2.rectangle(img, bb_s, bb_t, c, thickness, 1)
        score_text = f'{scores_nms[count]:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        thickness = 1
        box_w = bb_t[0] - bb_s[0]
        box_h = bb_t[1] - bb_s[1]
        org = (bb_s[0] + box_w // 4, bb_s[1] + box_h // 2)
        cv2.putText(img, score_text, org, font, fontScale, thickness=thickness, color=c, lineType=cv2.LINE_AA)
        count += 1

    #     print(f'{count} bb plotted')

    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # img = np.transpose(img, (2, 0, 1))
    # img = Image.fromarray(img)

    return img, nms_indices


def plot_bb_on_image_batch_from_masks_nms(mask_batch_tensor, img_batch_tensor, scores, iou_thresh=0.5, thickness=1,
                                          max_imgs=8, hard_thresh=None):
    # mask_batch_tensor: [batch_size, n_kp, 1, feature_dim, feature_dim]
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_bb = []
    indices = []
    for i in range(num_plot):
        img_np, nms_indices = plot_bb_on_image_from_masks_nms(mask_batch_tensor[i], img_batch_tensor[i], scores[i],
                                                              iou_thresh, thickness=thickness, hard_thresh=hard_thresh)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_bb.append(img_tensor.permute(2, 0, 1))
        indices.append(nms_indices)
    img_with_bb = torch.stack(img_with_bb, dim=0)
    return img_with_bb, indices


def plot_bb_on_image_from_masks(masks, image_tensor, thickness=1):
    # masks: [n_masks, 1, feature_dim, feature_dim]
    n_masks = masks.shape[0]
    mask_h, mask_w = masks.shape[2], masks.shape[3]
    height, width = image_tensor.size(1), image_tensor.size(2)

    img = transforms.ToPILImage()(image_tensor.cpu())

    img = np.array(img)
    cmap = color_map()
    cm = cmap[:n_masks].astype(int)
    count = 0
    for mask, color in zip(masks, cm):
        c = color.item(0), color.item(1), color.item(2)
        mask = mask.int().squeeze()  # [feature_dim, feature_dim]
        #         print(mask.shape)
        indices = (mask == 1).nonzero(as_tuple=False)
        #         print(indices.shape)
        if indices.shape[0] > 0:
            ws = (indices[0][1] * (width / mask_w) - thickness).clamp(0, width).int()
            wt = (indices[-1][1] * (width / mask_w) + thickness).clamp(0, width).int()
            hs = (indices[0][0] * (height / mask_h) - thickness).clamp(0, height).int()
            ht = (indices[-1][0] * (height / mask_h) + thickness).clamp(0, height).int()
            bb_s = (ws, hs)
            bb_t = (wt, ht)
            cv2.rectangle(img, bb_s, bb_t, c, thickness, 1)
            count += 1
    #     print(f'{count} bb plotted')

    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # img = np.transpose(img, (2, 0, 1))
    # img = Image.fromarray(img)

    return img


def plot_bb_on_image_batch_from_masks(mask_batch_tensor, img_batch_tensor, thickness=1, max_imgs=8):
    # mask_batch_tensor: [batch_size, n_kp, 1, feature_dim, feature_dim]
    num_plot = min(max_imgs, img_batch_tensor.shape[0])
    img_with_bb = []
    for i in range(num_plot):
        img_np = plot_bb_on_image_from_masks(mask_batch_tensor[i], img_batch_tensor[i], thickness=thickness)
        img_tensor = torch.tensor(img_np).float() / 255.0
        img_with_bb.append(img_tensor.permute(2, 0, 1))
    img_with_bb = torch.stack(img_with_bb, dim=0)
    return img_with_bb


def plot_masks_single(masks):
    # masks: [n_kp + 1, image_size, image_size]
    num_colors = masks.shape[0]
    masks = masks.argmax(0)
    palette = torch.tensor([2 ** 21 - 1, 2 ** 15 - 1, 2 ** 20 - 1])
    colors = torch.as_tensor([i for i in range(num_colors)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # cmap = color_map()
    # colors = cmap[:num_colors].astype(int)

    # plot the semantic segmentation predictions of n_kp + 1 classes in each color
    r = Image.fromarray(masks.byte().cpu().numpy())
    r.putpalette(colors)
    r = r.convert("RGB")
    r_t = transforms.ToTensor()(r)
    # r_t = r_t.repeat(3, 1, 1)
    # fig = plt.figure(figsize=(15, 15))
    # ax = fig.add_subplot(111)
    # ax.imshow(r_t.permute(1, 2, 0).data.cpu().numpy())
    # ax.set_axis_off()
    # plt.show()
    # r_t = transforms.ToTensor()(r)
    return r_t


def plot_masks_batch(masks_batch, max_imgs=8):
    # mask_batch_tensor: [batch_size, n_kp + 1, 1, feature_dim, feature_dim]
    masks_batch = masks_batch.squeeze(dim=2)
    num_plot = min(max_imgs, masks_batch.shape[0])
    img_with_mask = []
    for i in range(num_plot):
        img_tensor = plot_masks_single(masks_batch[i])
        img_with_mask.append(img_tensor)
    img_with_mask = torch.stack(img_with_mask, dim=0)
    return img_with_mask


def prepare_logdir(runname, src_dir='./'):
    td_prefix = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    dir_name = f'{td_prefix}_{runname}'
    path_to_dir = os.path.join(src_dir, dir_name)
    os.makedirs(path_to_dir, exist_ok=True)
    path_to_fig_dir = os.path.join(path_to_dir, 'figures')
    os.makedirs(path_to_fig_dir, exist_ok=True)
    path_to_save_dir = os.path.join(path_to_dir, 'saves')
    os.makedirs(path_to_save_dir, exist_ok=True)
    return path_to_dir


def save_config(src_dir, hparams):
    path_to_conf = os.path.join(src_dir, 'hparams.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(hparams, outfile, indent=2)


def log_line(src_dir, line):
    log_file = os.path.join(src_dir, 'log.txt')
    with open(log_file, 'a') as fp:
        fp.writelines(line)


def animate_trajectories(orig_trajectory, pred_trajectory, path='./traj_anim.gif'):
    total_images = []
    for i in range(len(orig_trajectory)):
        concat_img = np.concatenate([orig_trajectory[i],
                                     np.ones((orig_trajectory[i].shape[0], 4, orig_trajectory[i].shape[-1])),
                                     pred_trajectory[i].clip(0, 1)], axis=1)
        total_images.append((concat_img * 255).astype(np.uint8))
    imageio.mimsave(path, total_images, duration=2 / 50)  # 1/50


