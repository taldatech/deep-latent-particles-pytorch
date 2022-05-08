"""
Main DLP model neural network.
"""
# imports
import numpy as np
# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
# modules
from modules.modules import KeyPointCNNOriginal, VariationalKeyPointPatchEncoder, SpatialSoftmaxKP, SpatialLogSoftmaxKP, \
    ToGaussianMapHW, CNNDecoder, ObjectEncoder, ObjectDecoderCNN, PointNetPPToCNN
# util functions
from utils.util_func import reparameterize, get_kp_mask_from_gmap, create_masks_fast


class KeyPointVAE(nn.Module):
    def __init__(self, cdim=3, enc_channels=(16, 16, 32), prior_channels=(16, 16, 32), image_size=64, n_kp=1,
                 use_logsoftmax=False, pad_mode='replicate', sigma=0.1, dropout=0.0, dec_bone="gauss_pointnetpp",
                 patch_size=16, n_kp_enc=20, n_kp_prior=20, learned_feature_dim=16,
                 kp_range=(-1, 1), kp_activation="tanh", mask_threshold=0.2, anchor_s=0.25,
                 use_object_enc=False, use_object_dec=False, learn_order=False):
        super(KeyPointVAE, self).__init__()
        """
        cdim: channels of the input image (3...)
        enc_channels: channels for the posterior CNN (takes in the whole image)
        prior_channels: channels for prior CNN (takes in patches)
        n_kp: number of kp to extract from each (!) patch
        n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
        n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
        use_logsoftmax: for spatial-softmax, set True to use log-softmax for numerical stability
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        sigma: the prior std of the KP
        dropout: dropout for the CNNs. We don't use it though...
        dec_bone: decoder backbone -- "gauss_pointnetpp_feat": Masked Model, "gauss_pointnetpp": "Object Model
        patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
        kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
        mask_threshold: activation threshold (>thresh -> 1, else 0) for the binary mask created from the Gaussian-maps.
        anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
        learn_order: experimental feature to learn the order of keypoints - but it doesn't work yet.
        use_object_enc: set True to use a separate encoder to encode visual features of glimpses.
        use_object_dec: set True to use a separate decoder to decode glimpses (Object Model).
        """
        if dec_bone not in ["gauss_pointnetpp", "gauss_pointnetpp_feat"]:
            raise SystemError(f'unrecognized decoder backbone: {dec_bone}')
        print(f'decoder backbone: {dec_bone}')
        self.dec_bone = dec_bone
        self.image_size = image_size
        self.use_logsoftmax = use_logsoftmax
        self.sigma = sigma
        print(f'prior std: {self.sigma}')
        self.dropout = dropout
        self.kp_range = kp_range
        print(f'keypoints range: {self.kp_range}')
        self.num_patches = int((image_size // patch_size) ** 2)
        self.n_kp = n_kp
        self.n_kp_total = self.n_kp * self.num_patches
        self.n_kp_prior = min(self.n_kp_total, n_kp_prior)
        print(f'total number of kp: {self.n_kp_total} -> prior kp: {self.n_kp_prior}')
        self.n_kp_enc = n_kp_enc
        print(f'number of kp from encoder: {self.n_kp_enc}')
        self.kp_activation = kp_activation
        print(f'kp_activation: {self.kp_activation}')
        self.patch_size = patch_size
        self.features_dim = int(image_size // (2 ** (len(enc_channels) - 1)))
        self.learned_feature_dim = learned_feature_dim
        print(f'learnable feature dim: {learned_feature_dim}')
        self.mask_threshold = mask_threshold
        print(f'mask threshold: {self.mask_threshold}')
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        print(f'object patch size: {self.obj_patch_size}')
        self.use_object_enc = True if use_object_dec else use_object_enc
        self.use_object_dec = use_object_dec
        print(f'object encoder: {self.use_object_enc}, object decoder: {self.use_object_dec}')
        self.learn_order = learn_order
        print(f'learn particles order: {self.learn_order}')

        # encoder
        self.enc = KeyPointCNNOriginal(cdim=cdim, channels=enc_channels, image_size=image_size, n_kp=self.n_kp_enc,
                                       pad_mode=pad_mode, use_resblock=False)
        enc_output_dim = 2 * 2
        # flatten feature maps and extract statistics
        self.to_normal_stats = nn.Sequential(nn.Linear(self.n_kp_enc * self.features_dim ** 2, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, 128),
                                             nn.ReLU(True),
                                             nn.Linear(128, self.n_kp_enc * enc_output_dim))
        if self.use_object_dec:
            if self.learn_order:
                enc_aux_output_dim = 1 + self.n_kp_enc  # obj_on, ordering weights
            else:
                enc_aux_output_dim = 1  # obj_on
            self.aux_enc = nn.Sequential(nn.Linear(self.n_kp_enc * self.features_dim ** 2, 256),
                                         nn.ReLU(True),
                                         nn.Linear(256, 128),
                                         nn.ReLU(True),
                                         nn.Linear(128, self.n_kp_enc * enc_aux_output_dim))
        else:
            self.aux_enc = None
        # object encoder
        object_enc_output_dim = self.learned_feature_dim * 2  # [mu_features, sigma_features]
        self.object_enc = nn.Sequential(nn.Linear(self.n_kp_enc * self.features_dim ** 2, 256),
                                        nn.ReLU(True),
                                        nn.Linear(256, 128),
                                        nn.ReLU(True),
                                        nn.Linear(128, object_enc_output_dim))
        if self.use_object_enc:
            if self.use_object_dec:
                self.object_enc_sep = ObjectEncoder(z_dim=learned_feature_dim, anchor_size=anchor_s,
                                                    image_size=image_size, ch=cdim, margin=0, cnn=True)
            else:
                self.object_enc_sep = ObjectEncoder(z_dim=learned_feature_dim, anchor_size=anchor_s,
                                                    image_size=self.features_dim, ch=self.n_kp_enc,
                                                    margin=0, cnn=False, encode_location=True)
        else:
            self.object_enc_sep = None
        self.prior = VariationalKeyPointPatchEncoder(cdim=cdim, channels=prior_channels, image_size=image_size,
                                                     n_kp=n_kp, kp_range=self.kp_range,
                                                     patch_size=patch_size, use_logsoftmax=use_logsoftmax,
                                                     pad_mode=pad_mode, sigma=sigma, dropout=dropout,
                                                     learnable_logvar=False, learned_feature_dim=0)
        self.ssm = SpatialLogSoftmaxKP(kp_range=kp_range) if use_logsoftmax else SpatialSoftmaxKP(kp_range=kp_range)

        # decoder
        decoder_n_kp = 3 * self.n_kp_enc if self.dec_bone == "gauss_pointnetpp_feat" else 2 * self.n_kp_enc
        self.to_gauss_map = ToGaussianMapHW(sigma_w=sigma, sigma_h=sigma, kp_range=kp_range)
        self.pointnet = PointNetPPToCNN(axis_dim=2, target_hw=self.features_dim,
                                        n_kp=self.n_kp_enc, features_dim=self.learned_feature_dim,
                                        pad_mode=pad_mode)
        self.dec = CNNDecoder(cdim=cdim, channels=enc_channels, image_size=image_size, in_ch=decoder_n_kp,
                              n_kp=self.n_kp_enc + 1, pad_mode=pad_mode)
        # object decoder
        if self.use_object_dec:
            self.object_dec = ObjectDecoderCNN(patch_size=(self.obj_patch_size, self.obj_patch_size), num_chans=4,
                                               bottleneck_size=learned_feature_dim)
        else:
            self.object_dec = None
        self.init_weights()

    def get_parameters(self, prior=True, encoder=True, decoder=True):
        parameters = []
        if prior:
            parameters.extend(list(self.prior.parameters()))
        if encoder:
            parameters.extend(list(self.enc.parameters()))
            parameters.extend(list(self.to_normal_stats.parameters()))
            parameters.extend(list(self.object_enc.parameters()))
            if self.use_object_enc:
                parameters.extend(list(self.object_enc_sep.parameters()))
            if self.use_object_dec:
                parameters.extend(list(self.aux_enc.parameters()))
        if decoder:
            parameters.extend(list(self.dec.parameters()))
            parameters.extend(list(self.pointnet.parameters()))
            if self.use_object_dec:
                parameters.extend(list(self.object_dec.parameters()))
        return parameters

    def set_require_grad(self, prior_value=True, enc_value=True, dec_value=True):
        for param in self.prior.parameters():
            param.requires_grad = prior_value
        for param in self.enc.parameters():
            param.requires_grad = enc_value
        for param in self.to_normal_stats.parameters():
            param.requires_grad = enc_value
        for param in self.object_enc.parameters():
            param.requires_grad = enc_value
        if self.use_object_enc:
            for param in self.object_enc_sep.parameters():
                param.requires_grad = enc_value
        if self.use_object_dec:
            for param in self.aux_enc.parameters():
                param.requires_grad = enc_value
        for param in self.dec.parameters():
            param.requires_grad = dec_value
        for param in self.pointnet.parameters():
            param.requires_grad = dec_value
        if self.use_object_dec:
            for param in self.object_dec.parameters():
                param.requires_grad = dec_value

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass

    def encode(self, x, return_heatmap=False, mask=None):
        _, z_kp = self.enc(x)  # [batch_size, n_kp, features_dim, features_dim]
        if mask is None:
            masked_hm = z_kp
        else:
            masked_hm = mask * z_kp
        z_kp_v = masked_hm.view(masked_hm.shape[0], -1)  # [batch_size, n_kp * features_dim * features_dim]
        stats = self.to_normal_stats(z_kp_v)  # [batch_size, n_kp * 4]
        stats = stats.view(stats.shape[0], self.n_kp_enc, 2 * 2)
        # [batch_size, n_kp, 4 + learned_feature_dim * 2]
        mu_enc, logvar_enc = torch.chunk(stats, chunks=2, dim=-1)  # [batch_size, n_kp, 2 + learned_feature_dim]
        mu, logvar = mu_enc[:, :, :2], logvar_enc[:, :, :2]  # [x, y]

        logvar_p = torch.log(torch.tensor(self.sigma ** 2, device=logvar.device))
        if self.use_object_dec:
            stats_aux = self.aux_enc(z_kp_v.detach())
            # stats_aux = self.aux_enc(z_kp_v)
            if self.learn_order:
                stats_aux = stats_aux.view(stats_aux.shape[0], self.n_kp_enc, 1 + self.n_kp_enc)
                order_weights = stats_aux[:, :, 1:]
            else:
                stats_aux = stats_aux.view(stats_aux.shape[0], self.n_kp_enc, 1)
                order_weights = None
            mu_obj_weight = stats_aux[:, :, 0]
            mu_obj_weight = torch.sigmoid(mu_obj_weight)
        else:
            mu_obj_weight = None
            order_weights = None
        mu_features, logvar_features = None, None
        if self.kp_activation == "tanh":
            mu = torch.tanh(mu)
        elif self.kp_activation == "sigmoid":
            mu = torch.sigmoid(mu)

        mu = torch.cat([mu, torch.zeros_like(mu[:, 0]).unsqueeze(1)], dim=1)
        logvar = torch.cat([logvar, logvar_p * torch.ones_like(logvar[:, 0]).unsqueeze(1)], dim=1)

        if return_heatmap:
            return mu, logvar, z_kp, mu_features, logvar_features, mu_obj_weight, order_weights
        else:
            return mu, logvar, mu_features, logvar_features, mu_obj_weight, order_weights

    def encode_object_features(self, features_map, masks):
        # features_map [bs, n_kp, feature_dim, feature_dim]
        # masks: [bs, n_kp + 1, feature_dim, feature_dim]
        y = masks.unsqueeze(2) * features_map.unsqueeze(1)  # [bs, n_kp + 1, n_kp, feature_dim, feature_dim]
        y = y.view(y.shape[0], y.shape[1], -1)  # [bs, n_kp + 1, n_kp *  feature_dim ** 2]
        enc_out = self.object_enc(y)  # [bs, n_kp + 1, learned_feature_dim * 2]
        mu_features, logvar_features = torch.chunk(enc_out, chunks=2, dim=-1)  # [bs, n_kp + 1, learned_feature_dim]
        return mu_features, logvar_features

    def encode_object_features_sep(self, x, kp, features_map, masks):
        # x: [bs, ch, image_size, image_size]
        # kp :[bs, n_kp, 2]
        # features_map: [bs, n_kp, features_dim, features_dim]
        # masks: [bs, n_kp, features_dim, features_dim]

        batch_size, n_kp, features_dim, _ = masks.shape

        # object features
        obj_enc_out = self.object_enc_sep(x, kp.detach())
        mu_obj, logvar_obj, cropped_objects = obj_enc_out[0], obj_enc_out[1], obj_enc_out[2]
        if len(obj_enc_out) > 3:
            cropped_objects_masks = obj_enc_out[3]
        else:
            cropped_objects_masks = None

        # bg beatures
        if self.use_object_dec:
            obj_fmap_masks = create_masks_fast(kp.detach(), anchor_s=self.anchor_s, feature_dim=self.features_dim)
            bg_mask = 1 - obj_fmap_masks.squeeze(2).sum(1, keepdim=True).clamp(0,
                                                                               1)  # [bs, 1, features_dim, features_dim]
        else:
            bg_mask = masks[:, -1].unsqueeze(1)  # [bs, 1, features_dim, features_dim]
        masked_features = bg_mask.unsqueeze(2) * features_map.unsqueeze(1)  # [bs, 1, n_kp, f_dim, f_dim]
        masked_features = masked_features.view(batch_size, masked_features.shape[1], -1)  # flatten
        object_enc_out = self.object_enc(masked_features)  # [bs, 1, 2 * learned_features_dim]
        mu_bg, logvar_bg = object_enc_out.chunk(2, dim=-1)

        mu_features = torch.cat([mu_obj, mu_bg], dim=1)
        logvar_features = torch.cat([logvar_obj, logvar_bg], dim=1)

        return mu_features, logvar_features, cropped_objects, cropped_objects_masks

    def encode_all(self, x, return_heatmap=False, mask=None, deterministic=False):
        # posterior
        enc_out = self.encode(x, return_heatmap=True, mask=mask)
        mu, logvar, kp_heatmap, mu_features, logvar_features, obj_on, order_weights = enc_out
        if deterministic:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        gmap_1_fg = self.to_gauss_map(z[:, :-1], self.features_dim, self.features_dim)
        fg_masks_sep = get_kp_mask_from_gmap(gmap_1_fg, threshold=self.mask_threshold, binary=True,
                                             elementwise=True).detach()
        fg_masks = fg_masks_sep.sum(1, keepdim=True).clamp(0, 1)
        bg_masks = 1 - fg_masks
        masks_sep = torch.cat([fg_masks_sep, bg_masks], dim=1)

        if self.learned_feature_dim > 0:
            if self.use_object_enc:
                feat_source = x if self.use_object_dec else kp_heatmap.detach()
                obj_enc_out = self.encode_object_features_sep(feat_source, z[:, :-1], kp_heatmap.detach(),
                                                              masks_sep.detach())
                mu_features, logvar_features, cropped_objects = obj_enc_out[0], obj_enc_out[1], obj_enc_out[2]
            else:
                mu_features, logvar_features = self.encode_object_features(kp_heatmap.detach(), masks_sep)
        if return_heatmap:
            return mu, logvar, kp_heatmap, mu_features, logvar_features, obj_on, order_weights
        else:
            return mu, logvar, mu_features, logvar_features, obj_on, order_weights

    def encode_prior(self, x):
        return self.prior(x)

    def decode(self, z):
        return self.dec(z)

    def get_prior_kp(self, x, probs=False):
        _, z = self.encode_prior(x)
        return self.ssm(z, probs)

    def translate_patches(self, kp_batch, patches_batch, scale=None, translation=None):
        """
        translate patches to be centered around given keypoints
        kp_batch: [bs, n_kp, 2] in [-1, 1]
        patches: [bs, n_kp, ch_patches, patch_size, patch_size]
        scale: None or [bs, n_kp, 2] or [bs, n_kp, 1]
        translation: None or [bs, n_kp, 2] or [bs, n_kp, 1] (delta from kp)
        :return: translated_padded_pathces [bs, n_kp, ch, img_size, img_size]
        """
        batch_size, n_kp, ch_patch, patch_size, _ = patches_batch.shape
        img_size = self.image_size
        pad_size = (img_size - patch_size) // 2
        padded_patches_batch = F.pad(patches_batch, pad=[pad_size] * 4)
        delta_t_batch = 0.0 - kp_batch
        delta_t_batch = delta_t_batch.reshape(-1, delta_t_batch.shape[-1])  # [bs * n_kp, 2]
        padded_patches_batch = padded_patches_batch.reshape(-1, *padded_patches_batch.shape[2:])
        # [bs * n_kp, 3, patch_size, patch_size]
        zeros = torch.zeros([delta_t_batch.shape[0], 1], device=delta_t_batch.device).float()
        ones = torch.ones([delta_t_batch.shape[0], 1], device=delta_t_batch.device).float()

        if scale is None:
            scale_w = ones
            scale_h = ones
        elif scale.shape[-1] == 1:
            scale_w = scale[:, :-1].reshape(-1, scale.shape[-1])  # no need for bg kp
            scale_h = scale[:, :-1].reshape(-1, scale.shape[-1])  # no need for bg kp
        else:
            scale_h, scale_w = torch.split(scale[:, :-1], [1, 1], dim=-1)
            scale_w = scale_w.reshape(-1, scale_w.shape[-1])
            scale_h = scale_h.reshape(-1, scale_h.shape[-1])
        if translation is None:
            trans_w = zeros
            trans_h = zeros
        elif translation.shape[-1] == 1:
            trans_w = translation[:, :-1].reshape(-1, translation.shape[-1])  # no need for bg kp
            trans_h = translation[:, :-1].reshape(-1, translation.shape[-1])  # no need for bg kp
        else:
            trans_h, trans_w = torch.split(translation[:, :-1], [1, 1], dim=-1)
            trans_w = trans_w.reshape(-1, trans_w.shape[-1])
            trans_h = trans_h.reshape(-1, trans_h.shape[-1])

        theta = torch.cat([scale_h, zeros, delta_t_batch[:, 1].unsqueeze(-1) + trans_h,
                           zeros, scale_w, delta_t_batch[:, 0].unsqueeze(-1) + trans_w], dim=-1)

        theta = theta.view(-1, 2, 3)  # [batch_size * n_kp, 2, 3]
        align_corners = False
        padding_mode = 'zeros'
        # mode = "nearest"
        mode = 'bilinear'

        grid = F.affine_grid(theta, padded_patches_batch.size(), align_corners=align_corners)
        trans_padded_patches_batch = F.grid_sample(padded_patches_batch, grid, align_corners=align_corners,
                                                   mode=mode, padding_mode=padding_mode)

        trans_padded_patches_batch = trans_padded_patches_batch.view(batch_size, n_kp, *padded_patches_batch.shape[1:])
        # [bs, n_kp, ch, img_size, img_size]
        return trans_padded_patches_batch

    def get_objects_alpha_rgb(self, z_kp, z_features, scale=None, translation=None, deterministic=False,
                              order_weights=None):
        dec_objects = self.object_dec(z_features[:, :-1])  # [bs * n_kp, 4, patch_size, patch_size]
        dec_objects = dec_objects.view(-1, self.n_kp_enc,
                                       *dec_objects.shape[1:])  # [bs, n_kp, 4, patch_size, patch_size]
        # translate patches
        dec_objects_trans = self.translate_patches(z_kp[:, :-1], dec_objects, scale, translation)
        dec_objects_trans = dec_objects_trans.clamp(0, 1)  # STN can change values to be < 0
        # dec_objects_trans: [bs, n_kp, 3, im_size, im_size]
        if order_weights is not None:
            # for each particle, we get a one-hot vector of its place in the order
            # we then move all of its maps [4, h, w] to its new place via 1x1 grouped-convolution (group_size=batch_size)
            bs, n_kp, n_ch, h, w = dec_objects_trans.shape
            order_weights = order_weights.view(order_weights.shape[0], self.n_kp_enc, self.n_kp_enc, 1, 1)
            order_weights = F.gumbel_softmax(order_weights, hard=True, dim=1)  # straight-through gradients (hard=True)
            # order weights: [bs, n_kp, n_kp, 1, 1] - for each kp, its location in the order, in one-hot form
            # i.e., if kp 1 is 6 in the order of 8 kp, then its vector: [0 0 0 0 0 0 1 0]
            order_weights = order_weights.view(order_weights.shape[0] * self.n_kp_enc, self.n_kp_enc, 1, 1)
            reordered_objects = dec_objects_trans.reshape(1, -1, h * n_ch, w)  # [1, bs * n_kp, h * n_ch, w]
            ordered_objects = F.conv2d(reordered_objects, order_weights, bias=None, stride=1, groups=bs)
            ordered_objects = ordered_objects.view(bs, n_kp, n_ch, h, w)
            dec_objects_trans = ordered_objects

        # multiply by alpha channel
        a_obj, rgb_obj = torch.split(dec_objects_trans, [1, 3], dim=2)

        if not deterministic:
            attn_mask = self.to_gauss_map(z_kp[:, :-1], a_obj.shape[-1], a_obj.shape[-1]).unsqueeze(
                2).detach()
            a_obj = a_obj + self.sigma * torch.randn_like(a_obj) * attn_mask
        return dec_objects, a_obj, rgb_obj

    def stitch_objects(self, a_obj, rgb_obj, obj_on, bg, stitch_method='c'):
        # turn off inactive kp
        # obj_on: [bs, n_kp, 1]
        a_obj = obj_on[:, :, None, None, None] * a_obj  # [bs, n_kp, 4, im_size, im_size]
        if stitch_method == 'a':
            # layer-wise stitching, each particle is a layer
            # x_0 = bg
            # x_i = (1-a_i) * x_(i-1) + a_i * rgb_i
            rec = bg
            curr_mask = a_obj[:, 0]
            comp_masks = [curr_mask]  # to calculate the effective mask, only for plotting
            for i in range(a_obj.shape[1]):
                rec = (1 - a_obj[:, i]) * rec + a_obj[:, i] * rgb_obj[:, i]
                # rec = (1 - a_obj[:, i].detach()) * rec + a_obj[:, i] * rgb_obj[:, i]
                # what is the effect of this? bad, masks are not learned properly
                if i > 0:
                    available_space = 1.0 - curr_mask.detach()
                    curr_mask_tmp = torch.min(available_space, a_obj[:, i])
                    comp_masks.append(curr_mask_tmp)
                    curr_mask = curr_mask + curr_mask_tmp
            comp_masks = torch.stack(comp_masks, dim=1)
            dec_objects_trans = comp_masks * rgb_obj
            dec_objects_trans = dec_objects_trans.sum(1)
        elif stitch_method == 'b':
            # same formula as method 'a', but with detach and opening the recursive formula
            # x_n = bg * \prod_{i=1}^n (1-a_i) + a_n * rgb_n + a_(n-1) * rgb_(n-1) * (1-a_n) + ...
            # + a_1 * rgb_1 * \prod_{i=1}^{n-1} (1-a_i)
            bg_comp = torch.prod(1 - a_obj, dim=1) * bg
            obj = a_obj * rgb_obj
            # stitch
            rec = obj[:, -1]
            for i in reversed(range(a_obj.shape[1] - 1)):
                rec = rec + obj[:, i] * torch.prod((1 - a_obj[:, i + 1:].detach()), dim=1)
            dec_objects_trans = rec.detach()
            rec = rec + bg_comp
        else:
            # alpha-based stitching: we first calculate the effective masks, assuming the previous
            # masks already occupy some space that cannot be taken and finally we multiply the effective masks
            # by the rgb channel, the bg mask is the space left from the sum of all effective masks.
            curr_mask = a_obj[:, 0]
            comp_masks = [curr_mask]
            for i in range(1, a_obj.shape[1]):
                available_space = 1.0 - curr_mask.detach()  # also works
                curr_mask_tmp = torch.min(available_space, a_obj[:, i])
                comp_masks.append(curr_mask_tmp)
                curr_mask = curr_mask + curr_mask_tmp
            comp_masks = torch.stack(comp_masks, dim=1)
            comp_masks_sum = comp_masks.sum(1).clamp(0, 1)
            alpha_mask = 1.0 - comp_masks_sum
            dec_objects_trans = comp_masks * rgb_obj
            dec_objects_trans = dec_objects_trans.sum(1)  # [bs, 3, im_size, im_size]
            rec = alpha_mask * bg + dec_objects_trans
        return rec, dec_objects_trans

    def decode_objects(self, z_kp, z_features, obj_on, scale=None, translation=None, deterministic=False,
                       order_weights=None, bg=None):
        dec_objects, a_obj, rgb_obj = self.get_objects_alpha_rgb(z_kp, z_features, scale=scale, translation=translation,
                                                                 deterministic=deterministic,
                                                                 order_weights=order_weights)
        if bg is None:
            bg = torch.zeros_like(rgb_obj[:, 0])
        # stitching
        rec, dec_objects_trans = self.stitch_objects(a_obj, rgb_obj, obj_on=obj_on, bg=bg)
        return dec_objects, dec_objects_trans, rec

    def decode_all(self, z, z_features, kp_heatmap, obj_on, deterministic=False, order_weights=None):
        gmap_1_fg = self.to_gauss_map(z[:, :-1], self.features_dim, self.features_dim)
        gmap_1_bg = 1 - gmap_1_fg.sum(1, keepdim=True).clamp(0, 1).detach()
        gmap_1 = torch.cat([gmap_1_fg, gmap_1_bg], dim=1)
        fg_masks_sep = get_kp_mask_from_gmap(gmap_1_fg, threshold=self.mask_threshold, binary=True,
                                             elementwise=True).detach()
        fg_masks = fg_masks_sep.sum(1, keepdim=True).clamp(0, 1)
        bg_masks = 1 - fg_masks
        masks = torch.cat([fg_masks.expand_as(gmap_1_fg), bg_masks], dim=1)
        # decode object and translate them to the positions of the keypoints
        # decode
        z_features_in = z_features
        if self.dec_bone == "gauss_pointnetpp":
            if self.learned_feature_dim > 0:
                gmap_2 = self.pointnet(position=z.detach(),
                                       features=torch.cat([z.detach(), z_features_in], dim=-1))
            else:
                gmap_2 = self.pointnet(position=z.detach(), features=z.detach())
            gmap = torch.cat([gmap_1[:, :-1], gmap_2], dim=1)
        elif self.dec_bone == "gauss_pointnetpp_feat":
            if self.learned_feature_dim > 0:
                gmap_2 = self.pointnet(position=z.detach(),
                                       features=torch.cat([z.detach(), z_features_in], dim=-1))
            else:
                gmap_2 = self.pointnet(position=z.detach(), features=z.detach())

            fg_masks = masks[:, :-1]
            bg_masks = masks[:, -1].unsqueeze(1)
            gmap_2 = fg_masks * gmap_2
            gmap_3 = bg_masks * kp_heatmap.detach()
            gmap = torch.cat([gmap_1[:, :-1], gmap_2, gmap_3], dim=1)
        else:
            raise NotImplementedError('grow a dec bone')
        rec = self.dec(gmap)

        if z_features is not None and self.use_object_dec:
            object_dec_out = self.decode_objects(z, z_features, obj_on, deterministic=deterministic,
                                                 order_weights=order_weights, bg=rec)
            dec_objects, dec_objects_trans, rec = object_dec_out
        else:
            dec_objects_trans = None
            dec_objects = None
        return rec, dec_objects, dec_objects_trans

    def forward(self, x, deterministic=False, detach_decoder=False, x_prior=None, warmup=False, stg=False,
                noisy_masks=False):
        # stg: straight-through-gradients. not used.
        # first, extract prior KP proposals
        # prior
        if x_prior is None:
            x_prior = x
        kp_p = self.prior(x_prior, global_kp=True)
        kp_p = kp_p.view(x_prior.shape[0], -1, 2)  # [batch_size, n_kp_total, 2]
        # filter proposals by distance to the patches' center
        dist_from_center = self.prior.get_distance_from_patch_centers(kp_p, global_kp=True)
        _, indices = torch.topk(dist_from_center, k=self.n_kp_prior, dim=-1, largest=True)
        batch_indices = torch.arange(kp_p.shape[0]).view(-1, 1).to(kp_p.device)
        kp_p = kp_p[batch_indices, indices]
        # alternatively, just sample random kp
        # kp_p = kp_p[:, torch.randperm(kp_p.shape[1])[:self.n_kp_prior]]

        # encode posterior KP
        mu, logvar, kp_heatmap, mu_features, logvar_features, obj_on, order_weights = self.encode(x,
                                                                                                  return_heatmap=True)
        if deterministic:
            z = mu
        else:
            z = reparameterize(mu, logvar)

        # create gaussian maps (and masks) from the posterior keypoints
        gmap_1_fg = self.to_gauss_map(z[:, :-1], self.features_dim, self.features_dim)
        gmap_1_bg = 1 - gmap_1_fg.sum(1, keepdim=True).clamp(0, 1).detach()
        gmap_1 = torch.cat([gmap_1_fg, gmap_1_bg], dim=1)
        fg_masks_sep = get_kp_mask_from_gmap(gmap_1_fg, threshold=self.mask_threshold, binary=True,
                                             elementwise=True).detach()
        fg_masks = fg_masks_sep.sum(1, keepdim=True).clamp(0, 1)
        bg_masks = 1 - fg_masks
        masks = torch.cat([fg_masks.expand_as(gmap_1_fg), bg_masks], dim=1)
        masks_sep = torch.cat([fg_masks_sep, bg_masks], dim=1)

        # encode visual features
        if self.learned_feature_dim > 0:
            if self.use_object_enc:
                feat_source = x if self.use_object_dec else kp_heatmap.detach()
                obj_enc_out = self.encode_object_features_sep(feat_source, z[:, :-1], kp_heatmap.detach(),
                                                              masks_sep.detach())
                mu_features, logvar_features, cropped_objects = obj_enc_out[0], obj_enc_out[1], obj_enc_out[2]
                if len(obj_enc_out) > 3:
                    cropped_objects_masks = obj_enc_out[3]
                else:
                    cropped_objects_masks = None
            else:
                mu_features, logvar_features = self.encode_object_features(kp_heatmap.detach(), masks_sep)
                cropped_objects = None
                cropped_objects_masks = None

            if deterministic:
                z_features = mu_features
            else:
                z_features = reparameterize(mu_features, logvar_features)
        else:
            z_features = None
            cropped_objects = None
            cropped_objects_masks = None

        # decode
        if not warmup or not self.use_object_dec:
            z_features_fg, z_features_bg = torch.split(z_features, [self.n_kp_enc, 1], dim=1)
            z_features_in = torch.cat([z_features_fg.detach(), z_features_bg],
                                      dim=1) if self.use_object_dec else z_features
            if self.dec_bone == "gauss_pointnetpp":
                if self.learned_feature_dim > 0:
                    gmap_2 = self.pointnet(position=z.detach(),
                                           features=torch.cat([z.detach(), z_features_in], dim=-1))
                else:
                    gmap_2 = self.pointnet(position=z.detach(), features=z.detach())
                gmap = torch.cat([gmap_1[:, :-1], gmap_2], dim=1)
            elif self.dec_bone == "gauss_pointnetpp_feat":
                if self.learned_feature_dim > 0:
                    gmap_2 = self.pointnet(position=z.detach(),
                                           features=torch.cat([z.detach(), z_features_in], dim=-1))
                else:
                    gmap_2 = self.pointnet(position=z.detach(), features=z.detach())

                fg_masks = masks[:, :-1]
                bg_masks = masks[:, -1].unsqueeze(1)
                gmap_2 = fg_masks * gmap_2
                gmap_3 = bg_masks * kp_heatmap.detach()
                gmap = torch.cat([gmap_1[:, :-1], gmap_2, gmap_3], dim=1)
            else:
                raise NotImplementedError('grow a dec bone')
            if detach_decoder:
                rec = self.dec(gmap.detach())
            else:
                rec = self.dec(gmap)
        else:
            rec = torch.zeros_like(x)
            gmap = None

        # decode object and translate them to the positions of the keypoints
        if z_features is not None and self.use_object_dec:

            bern = torch.distributions.bernoulli.Bernoulli(probs=obj_on)
            sample_obj_on = bern.sample()  # [batch_size, n_kp, 1]
            sample_obj_on = sample_obj_on + obj_on - obj_on.detach()  # straight-through-gradient
            obj_on_in = sample_obj_on if stg else obj_on
            object_dec_out = self.decode_objects(z, z_features, obj_on_in, deterministic=not noisy_masks,
                                                 order_weights=order_weights, bg=rec)
            dec_objects, dec_objects_trans, rec = object_dec_out
        else:
            dec_objects_trans = None
            dec_objects = None
            gmap = None

        output_dict = {}
        output_dict['kp_p'] = kp_p
        output_dict['gmap'] = gmap
        output_dict['rec'] = rec
        output_dict['mu'] = mu
        output_dict['logvar'] = logvar
        output_dict['mu_features'] = mu_features
        output_dict['logvar_features'] = logvar_features
        # object stuff
        output_dict['cropped_objects_original'] = cropped_objects
        output_dict['cropped_objects_masks'] = cropped_objects_masks
        output_dict['obj_on'] = obj_on
        output_dict['dec_objects_original'] = dec_objects
        output_dict['dec_objects'] = dec_objects_trans
        output_dict['order_weights'] = order_weights

        return output_dict

    def lerp(self, other, betta):
        # weight interpolation for ema - not used in the paper
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = self.parameters()
            other_param = other.parameters()
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)
