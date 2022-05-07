# imports
import numpy as np
# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
from util_func import create_masks_fast
# torch geometric
# torch_geometric
from torch_geometric.nn import MessagePassing, global_max_pool, GCNConv, ResGatedGraphConv, radius_graph
from torch_geometric.nn import knn_graph as knn_graph_nn
from torch_cluster import knn_graph
from torch_cluster import fps


class ResidualBlock(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0, padding="zeros"):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None
        if padding == "zeros":
            self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1,
                                   groups=groups,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(nn.ReplicationPad2d(1),
                                       nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1,
                                                 padding=0, groups=groups, bias=False))
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        if padding == "zeros":
            self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1,
                                   groups=groups, bias=False)
        else:
            self.conv2 = nn.Sequential(nn.ReplicationPad2d(1),
                                       nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1,
                                                 padding=0, groups=groups, bias=False))
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        return output


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, pad=0, pool=False, upsample=False, bias=False,
                 activation=True, batchnorm=True, relu_type='leaky', pad_mode='zeros', use_resblock=False):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential()
        if use_resblock:
            self.main.add_module(f'conv_{c_in}_to_{c_out}', ResidualBlock(c_in, c_out, padding=pad_mode))
        else:
            if pad_mode != 'zeros':
                self.main.add_module('reflect_pad', nn.ReplicationPad2d(pad))
                pad = 0
            self.main.add_module(f'conv_{c_in}_to_{c_out}', nn.Conv2d(c_in, c_out, kernel_size,
                                                                      stride=stride, padding=pad, bias=bias))
        if batchnorm:
            self.main.add_module(f'bathcnorm_{c_out}', nn.BatchNorm2d(c_out))
        if activation:
            if relu_type == 'leaky':
                self.main.add_module(f'relu', nn.LeakyReLU(0.01))
            else:
                self.main.add_module(f'relu', nn.ReLU())
        if pool:
            self.main.add_module(f'max_pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        # if upsample:
        #     self.main.add_module(f'upsample_nearest_2',
        #                          nn.Upsample(scale_factor=2, mode='nearest'))
        if upsample:
            self.main.add_module(f'upsample_bilinear_2',
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            # TODO: bilinear with align_corners=False

    def forward(self, x):
        y = self.main(x)
        return y


class KeyPointCNN(nn.Module):
    def __init__(self, cdim=3, channels=(64, 128, 256, 512, 512, 512), image_size=64, n_kp=8, pad_mode='zeros',
                 dropout=0.0, use_resblock=False):
        super(KeyPointCNN, self).__init__()
        self.dropout = dropout
        self.cdim = cdim
        self.image_size = image_size
        self.n_kp = n_kp
        cc = channels[0]
        ch = cc
        self.main = nn.Sequential()
        self.main.add_module(f'in_block',
                             ConvBlock(cdim, cc, kernel_size=3, stride=1, pad=1, pool=True, pad_mode=pad_mode,
                                       use_resblock=use_resblock))

        sz = image_size
        for ch in channels[1:-1]:
            self.main.add_module('conv_in_{}'.format(sz), ConvBlock(cc, ch, kernel_size=3, stride=1, pad=1,
                                                                    pool=True, pad_mode=pad_mode,
                                                                    use_resblock=use_resblock))
            cc, sz = ch, sz // 2

        self.main.add_module(f'dropout', nn.Dropout2d(p=dropout))
        self.main.add_module(f'last_conv', ConvBlock(ch, channels[-1], kernel_size=3, stride=1, pad=1, pool=False
                                                     , pad_mode=pad_mode, use_resblock=use_resblock))
        self.keymap = nn.Sequential(nn.Conv2d(channels[-1], n_kp, kernel_size=1),
                                    nn.LeakyReLU(0.01, inplace=True))
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        # print("num fc features: ", num_fc_features)
        # self.fc = nn.Linear(num_fc_features, self.fc_output)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        y = self.main(x)
        # heatmap
        hm = self.keymap(y)
        return y, hm


class KeyPointCNNOriginal(nn.Module):
    def __init__(self, cdim=3, channels=(32, 64, 128, 256), image_size=64, n_kp=8, pad_mode='zeros',
                 use_resblock=False, first_conv_kernel_size=7):
        super(KeyPointCNNOriginal, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.n_kp = n_kp
        cc = channels[0]
        ch = cc
        first_conv_pad = first_conv_kernel_size // 2
        self.main = nn.Sequential()
        self.main.add_module(f'in_block_1',
                             ConvBlock(cdim, cc, kernel_size=first_conv_kernel_size, stride=1,
                                       pad=first_conv_pad, pool=False, pad_mode=pad_mode,
                                       use_resblock=use_resblock, relu_type='relu'))
        self.main.add_module(f'in_block_2',
                             ConvBlock(cc, cc, kernel_size=3, stride=1, pad=1, pool=False, pad_mode=pad_mode,
                                       use_resblock=use_resblock, relu_type='relu'))

        sz = image_size
        for ch in channels[1:]:
            self.main.add_module('conv_in_{}_0'.format(sz), ConvBlock(cc, ch, kernel_size=3, stride=2, pad=1,
                                                                      pool=False, pad_mode=pad_mode,
                                                                      use_resblock=use_resblock, relu_type='relu'))
            self.main.add_module('conv_in_{}_1'.format(ch), ConvBlock(ch, ch, kernel_size=3, stride=1, pad=1,
                                                                      pool=False, pad_mode=pad_mode,
                                                                      use_resblock=use_resblock, relu_type='relu'))
            cc, sz = ch, sz // 2

        # self.main.add_module(f'last_conv', ConvBlock(ch, channels[-1], kernel_size=3, stride=1, pad=1, pool=False
        #                                              , pad_mode=pad_mode, use_resblock=use_resblock))
        # self.keymap = nn.Sequential(nn.Conv2d(channels[-1], n_kp, kernel_size=1),
        #                             nn.ReLU(inplace=True))
        self.keymap = nn.Conv2d(channels[-1], n_kp, kernel_size=1)
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        # print("num fc features: ", num_fc_features)
        # self.fc = nn.Linear(num_fc_features, self.fc_output)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        y = self.main(x)
        # heatmap
        hm = self.keymap(y)
        return y, hm


class PointNetToCNN(nn.Module):
    def __init__(self, axis_dim=2, target_hw=16, n_kp=8, pad_mode='zeros'):
        super(PointNetToCNN, self).__init__()

        self.axis_dim = axis_dim
        self.n_kp = n_kp
        self.pointnet = nn.Sequential(
            nn.Conv1d(in_channels=axis_dim, out_channels=64, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        fc_out_dim = self.n_kp * 4 * 4
        self.fc = nn.Sequential(nn.Linear(512, 256, bias=True),
                                nn.ReLU(True),
                                nn.Linear(256, 128),
                                nn.ReLU(True),
                                nn.Linear(128, fc_out_dim),
                                nn.ReLU(True))

        num_upsample = int(np.log(target_hw) // np.log(2)) - 2
        print(f'pointnet to cnn num upsample: {num_upsample}')
        self.cnn = nn.Sequential()
        for i in range(num_upsample):
            self.cnn.add_module(f'depth_up_{i}', ConvBlock(n_kp, n_kp, kernel_size=3, pad=1,
                                                           upsample=True, pad_mode=pad_mode))

    def forward(self, x):
        # x [batch_size, n_kp, 2 or features_dim]
        x_in = x.transpose(2, 1)  # [batch_size, 2 or features_dim, n_kp]
        output = self.pointnet(x_in)
        output2 = output.max(dim=2)[0]
        lin_out = self.fc(output2)  # [batch_size, n_kp * 4 * 4]
        lin_out = lin_out.view(-1, self.n_kp, 4, 4)  # [batch_size, n_kp, 4, 4]
        cnn_out = self.cnn(lin_out)  # [batch_size, n_kp, target_hw, target_hw]
        return cnn_out


class STN(nn.Module):
    def __init__(self, axis_dim=2, n_kp=8):
        super(STN, self).__init__()

        self.axis_dim = axis_dim
        self.n_kp = n_kp
        self.pointnet = nn.Sequential(
            nn.Conv1d(in_channels=axis_dim, out_channels=64, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        fc_out_dim = self.n_kp * axis_dim
        self.fc = nn.Sequential(nn.Linear(512, 256, bias=True),
                                nn.ReLU(True),
                                nn.Linear(256, 128),
                                nn.ReLU(True),
                                nn.Linear(128, fc_out_dim))

    def forward(self, x):
        # x [batch_size, n_kp, 2]
        axis_dim = x.shape[-1]
        x_in = x.transpose(2, 1)  # [batch_size, 2, n_kp]
        output = self.pointnet(x_in)
        output2 = output.max(dim=2)[0]
        delta = self.fc(output2)  # [batch_size, n_kp * 2]
        delta = delta.view(-1, self.n_kp, axis_dim)  # [batch_size, n_kp, 2]
        x_new = x + delta
        return delta, x_new


class FCToCNN(nn.Module):
    def __init__(self, axis_dim=2, target_hw=16, n_kp=8, pad_mode='zeros'):
        super(FCToCNN, self).__init__()

        self.axis_dim = axis_dim
        self.n_kp = n_kp
        fc_out_dim = self.n_kp * 4 * 4
        self.fc = nn.Sequential(nn.Linear(axis_dim * n_kp, 128, bias=True),
                                nn.ReLU(True),
                                nn.Linear(128, 256),
                                nn.ReLU(True),
                                nn.Linear(256, fc_out_dim),
                                nn.ReLU(True))

        num_upsample = int(np.log(target_hw) // np.log(4))
        print(f'fc to cnn num upsample: {num_upsample}')
        self.cnn = nn.Sequential()
        for i in range(num_upsample):
            self.cnn.add_module(f'depth_up_{i}', ConvBlock(n_kp, n_kp, kernel_size=3, pad=1,
                                                           upsample=True, pad_mode=pad_mode))

    def forward(self, x):
        # x [batch_size, n_kp, 2 or features_dim]
        x = x.view(x.shape[0], -1)
        lin_out = self.fc(x)  # [batch_size, n_kp * 4 * 4]
        lin_out = lin_out.view(-1, self.n_kp, 4, 4)  # [batch_size, n_kp, 4, 4]
        cnn_out = self.cnn(lin_out)  # [batch_size, n_kp, target_hw, target_hw]
        return cnn_out


class PointNetToKP(nn.Module):
    def __init__(self, axis_dim=2, n_kp=8):
        super(PointNetToKP, self).__init__()

        self.axis_dim = axis_dim
        self.n_kp = n_kp
        self.pointnet = nn.Sequential(
            nn.Conv1d(in_channels=axis_dim, out_channels=64, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
        )

        fc_out_dim = self.n_kp * self.axis_dim * 2  # [mu, logvar]
        self.fc = nn.Sequential(nn.Linear(128, fc_out_dim, bias=True))

    def forward(self, x):
        # x [batch_size, n_kp, 2 or features_dim]
        x_in = x.transpose(2, 1)  # [batch_size, 2 or features_dim, n_kp]
        output = self.pointnet(x_in)
        output2 = output.max(dim=2)[0]
        lin_out = self.fc(output2)  # [batch_size, n_kp * axis_dim * 2]
        lin_out = lin_out.view(-1, self.n_kp, self.axis_dim * 2)  # [batch_size, n_kp, 2 * axis_dim]
        mu, logvar = torch.chunk(lin_out, 2, dim=-1)  # [batch_size, n_kp, axis_dim]
        return mu, logvar


# coverting to probabilities
class SpatialSoftmaxKP(torch.nn.Module):
    def __init__(self, kp_range=(0, 1)):
        super().__init__()
        self.kp_range = kp_range

    def forward(self, heatmap, probs=False):
        batch_size, n_kp, height, width = heatmap.shape
        # marginalize over height (y)
        s_h = torch.mean(heatmap, dim=3)  # [batch_size, n_kp, features_dim_height]
        sm_h = torch.softmax(s_h, dim=-1)  # [batch_size, n_kp, features_dim_height]
        # marginalize over width (x)
        s_w = torch.mean(heatmap, dim=2)  # [batch_size, n_kp, features_dim_width]
        # probability per spatial coordinate
        sm_w = torch.softmax(s_w, dim=-1)  # [batch_size, n_kp, features_dim_width]
        # each coordinate [0, 1] is assigned a probability
        y_axis = torch.linspace(self.kp_range[0], self.kp_range[1], height).type_as(sm_h).expand(1, 1, -1).to(
            sm_h.device)  # [1, 1, features_dim_height]
        # expected value: proability per coordinate * coordinate
        kp_h = torch.sum(sm_h * y_axis, dim=-1, keepdim=True)  # [batch_size, n_kp, 1]
        kp_h = kp_h.squeeze(-1)  # [batch_size, n_kp], y coordinate of each kp

        x_axis = torch.linspace(self.kp_range[0], self.kp_range[1], width).type_as(sm_w).expand(1, 1, -1).to(
            sm_w.device)  # [1, 1, features_dim_width]
        kp_w = torch.sum(sm_w * x_axis, dim=-1, keepdim=True).squeeze(-1)  # [batch_size, n_kp], x coordinate of each kp

        # stack keypoints
        kp = torch.stack([kp_h, kp_w], dim=-1)  # [batch_size, n_kp, 2], x, y coordinates of each kp

        if probs:
            return kp, sm_h, sm_w
        else:
            return kp


class SpatialLogSoftmaxKP(torch.nn.Module):
    def __init__(self, kp_range=(0, 1)):
        super().__init__()
        self.kp_range = kp_range

    def forward(self, heatmap, probs=False):
        batch_size, n_kp, height, width = heatmap.shape
        # marginalize over height (y)
        s_h = torch.mean(heatmap, dim=3)  # [batch_size, n_kp, features_dim_height]
        sm_h = torch.log_softmax(s_h, dim=-1)  # [batch_size, n_kp, features_dim_height]
        # marginalize over width (x)
        s_w = torch.mean(heatmap, dim=2)  # [batch_size, n_kp, features_dim_width]
        sm_w = torch.log_softmax(s_w, dim=-1)  # [batch_size, n_kp, features_dim_width]
        # each coordinate [0, 1] is assigned a probability
        y_axis = torch.log(torch.linspace(self.kp_range[0], self.kp_range[1], height)).type_as(sm_h).expand(1, 1,
                                                                                                            -1).to(
            sm_h.device)
        # [1, 1, features_dim_height]
        # expected value: proability per coordinate * coordinate
        kp_h = torch.sum(torch.exp(sm_h + y_axis), dim=-1, keepdim=True)  # [batch_size, n_kp, 1]
        kp_h = kp_h.squeeze(-1)  # [batch_size, n_kp], y coordinate of each kp

        x_axis = torch.log(torch.linspace(self.kp_range[0], self.kp_range[1], width)).type_as(sm_w).expand(1, 1, -1).to(
            sm_w.device)
        # [1, 1, features_dim_width]
        kp_w = torch.sum(torch.exp(sm_w + x_axis), dim=-1, keepdim=True).squeeze(-1)
        # [batch_size, n_kp], x coordinate of each kp

        # stack keypoints
        kp = torch.stack([kp_h, kp_w], dim=-1)  # [batch_size, n_kp, 2], x, y coordinates of each kp

        if probs:
            return kp, sm_h, sm_w
        else:
            return kp


# converting to a gaussian map centered at the coordinates of the keypoint
class ToGaussianMap(nn.Module):
    def __init__(self, sigma=0.1, eps=1e-6):
        super().__init__()
        self.sigma = sigma
        self.eps = eps

    def forward(self, kp, height, width):
        batch_size, n_kp, _ = kp.shape
        # get means
        h_mean, w_mean = kp[:, :, 0], kp[:, :, 1]
        # create a coordinate map for each axis
        h_map = torch.linspace(0, 1, height, device=h_mean.device).type_as(h_mean)  # [height]
        w_map = torch.linspace(0, 1, width, device=w_mean.device).type_as(w_mean)  # [width]
        # duplicate for all keypoints in the batch
        h_map = h_map.expand(batch_size, n_kp, height)  # [batch_size, n_kp, height]
        w_map = w_map.expand(batch_size, n_kp, width)  # [batch_size, n_kp, width]
        # repeat the mean to match dimensions
        h_mean_m = h_mean.expand(height, -1, -1)  # [height, batch_size, n_kp]
        h_mean_m = h_mean_m.permute(1, 2, 0)  # [batch_size, n_kp, height]
        w_mean_m = w_mean.expand(width, -1, -1)  # [width, batch_size, n_kp]
        w_mean_m = w_mean_m.permute(1, 2, 0)  # [batch_size, n_kp, width]
        # for each pixel in the map, calculate the squared distance from the mean
        h_sdiff = (h_map - h_mean_m) ** 2  # [batch_size, n_kp, height]
        w_sdiff = (w_map - w_mean_m) ** 2  # [batch_size, n_kp, width]
        # compute gaussian
        # duplicate for the other dimension
        hm = h_sdiff.expand(width, -1, -1, -1).permute(1, 2, 3, 0)  # [batch_size, n_kp, height, width]
        wm = w_sdiff.expand(height, -1, -1, -1).permute(1, 2, 0, 3)  # [batch_size, n_kp, height, width]
        #         gm = - (hm + wm + self.eps).sqrt_() / (2 * self.sigma ** 2) # WHY sqrt?
        gm = - (hm + wm) / (2 * self.sigma ** 2)
        gm = torch.exp(gm)  # [batch_size, n_kp, height, width]
        return gm


class ToGaussianMapHW(nn.Module):
    def __init__(self, sigma_w=0.1, sigma_h=0.1, kp_range=(0, 1)):
        super().__init__()
        self.sigma_w = sigma_w
        self.sigma_h = sigma_h
        self.kp_range = kp_range

    def forward(self, kp, height, width, logvar_h=None, logvar_w=None):
        batch_size, n_kp, _ = kp.shape
        # get means
        h_mean, w_mean = kp[:, :, 0], kp[:, :, 1]
        # create a coordinate map for each axis
        h_map = torch.linspace(self.kp_range[0], self.kp_range[1], height, device=h_mean.device).type_as(
            h_mean)  # [height]
        w_map = torch.linspace(self.kp_range[0], self.kp_range[1], width, device=w_mean.device).type_as(
            w_mean)  # [width]
        # duplicate for all keypoints in the batch
        h_map = h_map.expand(batch_size, n_kp, height)  # [batch_size, n_kp, height]
        w_map = w_map.expand(batch_size, n_kp, width)  # [batch_size, n_kp, width]
        # repeat the mean to match dimensions
        h_mean_m = h_mean.expand(height, -1, -1)  # [height, batch_size, n_kp]
        h_mean_m = h_mean_m.permute(1, 2, 0)  # [batch_size, n_kp, height]
        w_mean_m = w_mean.expand(width, -1, -1)  # [width, batch_size, n_kp]
        w_mean_m = w_mean_m.permute(1, 2, 0)  # [batch_size, n_kp, width]
        # for each pixel in the map, calculate the squared distance from the mean
        h_sdiff = (h_map - h_mean_m) ** 2  # [batch_size, n_kp, height]
        w_sdiff = (w_map - w_mean_m) ** 2  # [batch_size, n_kp, width]
        # compute gaussian
        # duplicate for the other dimension
        hm = h_sdiff.expand(width, -1, -1, -1).permute(1, 2, 3, 0)  # [batch_size, n_kp, height, width]
        wm = w_sdiff.expand(height, -1, -1, -1).permute(1, 2, 0, 3)  # [batch_size, n_kp, height, width]
        #         gm = - (hm + wm + self.eps).sqrt_() / (2 * self.sigma ** 2) # WHY sqrt?
        #         gm = - (hm + wm) / (2 * self.sigma ** 2)
        if logvar_h is not None:
            sigma_h = torch.exp(0.5 * logvar_h)
            sigma_h = sigma_h.expand(height, -1, -1)  # [height, batch_size, n_kp]
            sigma_h = sigma_h.permute(1, 2, 0)  # [batch_size, n_kp, height]
            sigma_h = sigma_h.expand(width, -1, -1, -1).permute(1, 2, 3, 0)  # [batch_size, n_kp, height, width]
        else:
            sigma_h = self.sigma_h
        if logvar_w is not None:
            sigma_w = torch.exp(0.5 * logvar_w)
            sigma_w = sigma_w.expand(width, -1, -1)  # [width, batch_size, n_kp]
            sigma_w = sigma_w.permute(1, 2, 0)  # [batch_size, n_kp, width]
            sigma_w = sigma_w.expand(height, -1, -1, -1).permute(1, 2, 0, 3)  # [batch_size, n_kp, height, width]
        else:
            sigma_w = self.sigma_w
        #         print(hm.shape, sigma_h.shape, wm.shape, sigma_w.shape)
        gm = -0.5 * (hm / (sigma_h ** 2) + wm / (sigma_w ** 2))
        gm = torch.exp(gm)  # [batch_size, n_kp, height, width]
        return gm


# class CNNDecoder(nn.Module):
#     def __init__(self, cdim=3, channels=(64, 128, 256, 512, 512, 512), image_size=64, in_ch=16, n_kp=8,
#                  pad_mode='zeros', use_resblock=False):
#         super(CNNDecoder, self).__init__()
#         self.cdim = cdim
#         self.image_size = image_size
#         cc = channels[-1]
#         self.in_ch = in_ch
#         self.n_kp = n_kp
#
#         sz = 4
#
#         self.main = nn.Sequential()
#         self.main.add_module('depth_up',
#                              ConvBlock(self.in_ch, cc, kernel_size=3, pad=1, upsample=True, pad_mode=pad_mode,
#                                        use_resblock=use_resblock))
#         for ch in reversed(channels[1:-1]):
#             self.main.add_module('conv_to_{}'.format(sz * 2), ConvBlock(cc, ch, kernel_size=3, pad=1, upsample=True,
#                                                                         pad_mode=pad_mode, use_resblock=use_resblock))
#             cc, sz = ch, sz * 2
#
#         self.main.add_module('conv_to_{}'.format(sz * 2),
#                              ConvBlock(cc, self.n_kp * channels[0], kernel_size=3, pad=1, upsample=False,
#                                        pad_mode=pad_mode, use_resblock=use_resblock))
#         self.final_conv = ConvBlock(self.n_kp * channels[0], cdim, kernel_size=1, bias=True,
#                                     activation=False, batchnorm=False)
#
#     def forward(self, z, masks=None):
#         y = self.main(z)
#         if masks is not None:
#             # masks: [bs, n_kp, feat_dim, feat_dim]
#             bs, n_kp, fs, _ = masks.shape
#             # y: [bs, n_kp * ch[0], feat_dim, feat_dim]
#             y = y.view(bs, n_kp, -1, fs, fs)
#             y = masks.unsqueeze(2) * y
#             y = y.view(bs, -1, fs, fs)
#         y = self.final_conv(y)
#         return y

class CNNDecoder(nn.Module):
    def __init__(self, cdim=3, channels=(64, 128, 256, 512, 512, 512), image_size=64, in_ch=16, n_kp=8,
                 pad_mode='zeros', use_resblock=False):
        super(CNNDecoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[-1]
        self.in_ch = in_ch
        self.n_kp = n_kp

        sz = 4

        self.main = nn.Sequential()
        self.main.add_module('depth_up',
                             ConvBlock(self.in_ch, cc, kernel_size=3, pad=1, upsample=True, pad_mode=pad_mode,
                                       use_resblock=use_resblock))
        for ch in reversed(channels[1:-1]):
            self.main.add_module('conv_to_{}'.format(sz * 2), ConvBlock(cc, ch, kernel_size=3, pad=1, upsample=True,
                                                                        pad_mode=pad_mode, use_resblock=use_resblock))
            cc, sz = ch, sz * 2

        self.main.add_module('conv_to_{}'.format(sz * 2),
                             ConvBlock(cc, self.n_kp * (channels[0] // self.n_kp + 1), kernel_size=3, pad=1,
                                       upsample=False,
                                       pad_mode=pad_mode, use_resblock=use_resblock))
        self.final_conv = ConvBlock(self.n_kp * (channels[0] // self.n_kp + 1), cdim, kernel_size=1, bias=True,
                                    activation=False, batchnorm=False)

    def forward(self, z, masks=None):
        y = self.main(z)
        if masks is not None:
            # masks: [bs, n_kp, feat_dim, feat_dim]
            bs, n_kp, fs, _ = masks.shape
            # y: [bs, n_kp * ch[0], feat_dim, feat_dim]
            y = y.view(bs, n_kp, -1, fs, fs)
            y = masks.unsqueeze(2) * y
            y = y.view(bs, -1, fs, fs)
        y = self.final_conv(y)
        return y


class ImagePatcher(nn.Module):
    def __init__(self, cdim=3, image_size=64, patch_size=16):
        super(ImagePatcher, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.patch_size = patch_size
        self.kh, self.kw = self.patch_size, self.patch_size  # kernel size
        self.dh, self.dw = self.patch_size, patch_size  # stride
        self.unfold_shape = self.get_unfold_shape()
        self.patch_location_idx = self.get_patch_location_idx()
        # print(f'unfold shape: {self.unfold_shape}')
        # print(f'patch locations: {self.patch_location_idx}')

    def get_patch_location_idx(self):
        h = np.arange(0, self.image_size)[::self.patch_size]
        w = np.arange(0, self.image_size)[::self.patch_size]
        ww, hh = np.meshgrid(h, w)
        hw = np.stack((hh, ww), axis=-1)
        hw = hw.reshape(-1, 2)
        return torch.from_numpy(hw).int()

    def get_patch_centers(self):
        mid = self.patch_size // 2
        patch_locations_idx = self.get_patch_location_idx()
        patch_locations_idx += mid
        #         patch_locations_idx += torch.from_numpy(np.array([mid, mid])).int()
        return patch_locations_idx

    def get_unfold_shape(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        patches = dummy_input.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        unfold_shape = patches.shape[1:]
        return unfold_shape

    def img_to_patches(self, x):
        patches = x.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        patches = patches.contiguous().view(patches.shape[0], patches.shape[1], -1, self.kh, self.kw)
        return patches

    def patches_to_img(self, x):
        patches_orig = x.view(x.shape[0], *self.unfold_shape)
        output_h = self.unfold_shape[1] * self.unfold_shape[2]
        output_w = self.unfold_shape[2] * self.unfold_shape[4]
        patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_orig = patches_orig.view(-1, self.cdim, output_h, output_w)
        return patches_orig

    def forward(self, x, patches=True):
        # x [batch_size, 3, image_size, image_size] or [batch_size, 3, num_patches, image_size, image_size]
        if patches:
            return self.img_to_patches(x)
        else:
            return self.patches_to_img(x)


class VariationalKeyPointPatchEncoder(nn.Module):
    def __init__(self, cdim=3, channels=(16, 16, 32), image_size=64, n_kp=4, patch_size=16, kp_range=(0, 1),
                 use_logsoftmax=False, pad_mode='reflect', sigma=0.1, dropout=0.0, learnable_logvar=False,
                 learned_feature_dim=0):
        super(VariationalKeyPointPatchEncoder, self).__init__()
        self.use_logsoftmax = use_logsoftmax
        self.image_size = image_size
        self.dropout = dropout
        self.kp_range = kp_range
        self.n_kp = n_kp  # kp per patch
        self.patcher = ImagePatcher(cdim=cdim, image_size=image_size, patch_size=patch_size)
        self.features_dim = int(patch_size // (2 ** (len(channels) - 1)))
        # self.enc = KeyPointCNN(cdim=cdim, channels=channels, image_size=patch_size, n_kp=n_kp, pad_mode=pad_mode,
        #                        dropout=dropout, use_resblock=False)
        self.enc = KeyPointCNNOriginal(cdim=cdim, channels=channels, image_size=patch_size, n_kp=n_kp,
                                       pad_mode=pad_mode, use_resblock=False, first_conv_kernel_size=3)
        self.ssm = SpatialLogSoftmaxKP(kp_range=kp_range) if use_logsoftmax else SpatialSoftmaxKP(kp_range=kp_range)
        # self.to_gauss_map = ToGaussianMapHW(sigma_w=sigma, sigma_h=sigma, kp_range=kp_range)
        self.sigma = sigma
        self.learnable_logvar = learnable_logvar
        self.learned_feature_dim = learned_feature_dim
        if self.learnable_logvar:
            self.to_logvar = nn.Sequential(nn.Linear(self.n_kp * (self.features_dim ** 2), 512),
                                           nn.ReLU(True),
                                           nn.Linear(512, 256),
                                           nn.ReLU(True),
                                           nn.Linear(256, self.n_kp * 2))  # logvar_x, logvar_y
        if self.learned_feature_dim > 0:
            self.to_features = nn.Sequential(nn.Linear(self.n_kp * (self.features_dim ** 2), 512),
                                             nn.ReLU(True),
                                             nn.Linear(512, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, self.n_kp * self.learned_feature_dim))  # logvar_x, logvar_y

    def img_to_patches(self, x):
        return self.patcher.img_to_patches(x)

    def patches_to_img(self, x):
        return self.patcher.patches_to_img(x)

    def get_global_kp(self, local_kp):
        # local_kp: [batch_size, num_patches, n_kp, 2]
        batch_size, num_patches, n_kp, _ = local_kp.shape
        global_coor = self.patcher.get_patch_location_idx().to(local_kp.device)  # [num_patches, 2]
        global_coor = global_coor[:, None, :].repeat(1, n_kp, 1)
        #         print(f'global_coor shape: {global_coor.shape}')
        # global_coor = (local_kp * (self.patcher.patch_size - 1) + global_coor) / (self.image_size - 1)
        global_coor = (((local_kp - self.kp_range[0]) / (self.kp_range[1] - self.kp_range[0])) * (
                self.patcher.patch_size - 1) + global_coor) / (self.image_size - 1)
        global_coor = global_coor * (self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        return global_coor

    def get_distance_from_patch_centers(self, kp, global_kp=False):
        if not global_kp:
            global_coor = self.get_global_kp(kp).view(kp.shape[0], -1, 2)
        else:
            global_coor = kp
        # centers = 0.5 * torch.ones_like(kp).to(kp.device)
        centers = 0.5 * (self.kp_range[1] + self.kp_range[0]) * torch.ones_like(kp).to(kp.device)
        global_centers = self.get_global_kp(centers.view(kp.shape[0], -1, self.n_kp, 2)).view(kp.shape[0], -1, 2)
        return ((global_coor - global_centers) ** 2).sum(-1)

    def encode(self, x, global_kp=False):
        # x: [batch_size, cdim, image_size, image_size]
        batch_size, cdim, image_size, image_size = x.shape
        x_patches = self.img_to_patches(x)  # [batch_size, cdim, num_patches, patch_size, patch_size]
        #         print(f'x_patches shape: {x_patches.shape}')
        x_patches = x_patches.permute(0, 2, 1, 3, 4)  # [batch_size, num_patches, cdim, patch_size, patch_size]
        #         print(f'x_patches shape: {x_patches.shape}')
        x_patches = x_patches.contiguous().view(-1, cdim, self.patcher.patch_size, self.patcher.patch_size)
        #         print(f'x_patches shape: {x_patches.shape}')
        # [batch_size*num_patches, cdim, patch_size, patch_size]
        #         logvar_kp = torch.log(torch.tensor(self.sigma ** 2).float())
        _, z = self.enc(x_patches)  # [batch_size*num_patches, n_kp, features_dim, features_dim]
        #         print(f'z shape: {z.shape}')
        mu_kp = self.ssm(z, probs=False)  # [batch_size * num_patches, n_kp, 2]
        mu_kp = mu_kp.view(batch_size, -1, self.n_kp, 2)  # [batch_size, num_patches, n_kp, 2]
        if global_kp:
            mu_kp = self.get_global_kp(mu_kp)
        if self.learned_feature_dim > 0:
            mu_features = self.to_features(z.view(z.shape[0], -1))
            mu_features = mu_features.view(batch_size, -1, self.n_kp, self.learned_feature_dim)
            # [batch_size, num_patches, n_kp, learned_feature_dim]
        if self.learnable_logvar:
            logvar_kp = self.to_logvar(z.view(z.shape[0], -1))
            logvar_kp = logvar_kp.view(batch_size, -1, self.n_kp, 2)  # [batch_size, num_patches, n_kp, 2]
            if self.learned_feature_dim > 0:
                return mu_kp, logvar_kp, mu_features
            else:
                return mu_kp, logvar_kp
        elif self.learned_feature_dim > 0:
            return mu_kp, mu_features
        else:
            return mu_kp

    def forward(self, x, global_kp=False):
        return self.encode(x, global_kp)


class ObjectEncoder(nn.Module):
    def __init__(self, z_dim, anchor_size, image_size, cnn_channels=(16, 16, 32), margin=0, ch=3, cnn=False,
                 encode_location=False):
        super().__init__()

        self.anchor_size = anchor_size
        self.channels = cnn_channels
        self.z_dim = z_dim
        self.image_size = image_size
        self.patch_size = np.round(anchor_size * (image_size - 1)).astype(int)
        self.margin = margin
        self.crop_size = self.patch_size + 2 * margin
        self.ch = ch
        self.encode_location = encode_location

        if cnn:
            self.cnn = KeyPointCNNOriginal(cdim=ch, channels=cnn_channels, image_size=self.crop_size, n_kp=32,
                                           pad_mode='reflect', use_resblock=False, first_conv_kernel_size=3)
        else:
            self.cnn = None
        fc_in_dim = 32 * ((self.crop_size // 4) ** 2) if cnn else self.ch * (self.crop_size ** 2)
        fc_in_dim = fc_in_dim + 2 if self.encode_location else fc_in_dim
        self.fc = nn.Sequential(nn.Linear(fc_in_dim, 256),
                                nn.ReLU(True),
                                nn.Linear(256, 128),
                                nn.ReLU(True),
                                nn.Linear(128, self.z_dim * 2))

    def center_objects(self, kp, padded_objects):
        batch_size, n_kp, _ = kp.shape
        delta_tr_batch = kp
        delta_tr_batch = delta_tr_batch.reshape(-1, delta_tr_batch.shape[-1])  # [bs * n_kp, 2]
        source_batch = padded_objects.reshape(-1, *padded_objects.shape[2:])

        zeros = torch.zeros([delta_tr_batch.shape[0], 1], device=delta_tr_batch.device).float()
        ones = torch.ones([delta_tr_batch.shape[0], 1], device=delta_tr_batch.device).float()

        theta = torch.cat([ones, zeros, delta_tr_batch[:, 1].unsqueeze(-1),
                           zeros, ones, delta_tr_batch[:, 0].unsqueeze(-1)], dim=-1)
        theta = theta.view(-1, 2, 3)  # [batch_size * n_kp, 2, 3]

        align_corners = False
        padding_mode = 'zeros'
        # mode = "nearest"
        mode = 'bilinear'

        grid = F.affine_grid(theta, source_batch.size(), align_corners=align_corners)
        trans_source_batch = F.grid_sample(source_batch, grid, align_corners=align_corners,
                                           mode=mode, padding_mode=padding_mode)
        trans_source_batch = trans_source_batch.view(batch_size, n_kp, *source_batch.shape[1:])
        # debug
        # sample_object = trans_source_batch[0, 0].permute(1, 2, 0).clamp(0, 1).data.cpu().numpy()
        # sample_object = padded_objects[0, 0].permute(1, 2, 0).clamp(0, 1).data.cpu().numpy()
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111)
        # ax.imshow(sample_object)
        # ax.set_axis_off()
        # plt.show()
        return trans_source_batch

    def get_cropped_objects(self, centered_objects):
        center_idx = self.image_size // 2
        margin = self.margin
        w_start = center_idx - self.patch_size // 2 - margin
        w_end = center_idx + self.patch_size // 2 + margin
        h_start = center_idx - self.patch_size // 2 - margin
        h_end = center_idx + self.patch_size // 2 + margin
        cropped_objects = centered_objects[:, :, :, w_start:w_end, h_start:h_end]
        return cropped_objects

    def forward(self, x, kp):
        # x: [bs, ch, image_size, image_size]
        # kp: [bs, n_kp, 2] in [-1, 1]
        batch_size, _, _, img_size = x.shape
        _, n_kp, _ = kp.shape
        # create masks from kp
        masks = create_masks_fast(kp.detach(), self.anchor_size, feature_dim=self.image_size)
        # [batch_size, n_kp, 1, feature_dim, feature_dim]
        # extract objects
        padded_objects = masks * x.unsqueeze(1)  # [batch_size, n_kp, ch, image_size, image_size]
        # center objects
        centered_objects = self.center_objects(kp, padded_objects)  # [batch_size, n_kp, ch, image_size, image_size]
        # get crop
        cropped_objects = self.get_cropped_objects(centered_objects)
        # [batch_size, n_kp, ch, patch_size + margin * 2, image_size + margin * 2]

        # encode objects - fc
        if self.cnn is not None:
            _, cropped_objects_cnn = self.cnn(cropped_objects.view(-1, *cropped_objects.shape[2:]))
        else:
            cropped_objects_cnn = cropped_objects
        cropped_objects_flat = cropped_objects_cnn.reshape(batch_size, n_kp, -1)  # flatten
        if self.encode_location:
            cropped_objects_flat = torch.cat([cropped_objects_flat, kp], dim=-1)  # [batch_size, n_kp, .. + 2]
        enc_out = self.fc(cropped_objects_flat)
        mu, logvar = enc_out.chunk(2, dim=-1)  # [batch_size, n_kp, z_dim]
        return mu, logvar, cropped_objects


class ObjectDecoderCNN(nn.Module):
    def __init__(self, patch_size, num_chans=4, bottleneck_size=128, pad_mode='reflect'):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.num_chans = num_chans

        self.in_ch = 32

        fc_out_dim = self.in_ch * 8 * 8
        self.fc = nn.Sequential(nn.Linear(bottleneck_size, 256, bias=True),
                                nn.ReLU(True),
                                nn.Linear(256, fc_out_dim),
                                nn.ReLU(True))

        num_upsample = int(np.log(patch_size[0]) // np.log(2)) - 3
        print(f'ObjDecCNN: fc to cnn num upsample: {num_upsample}')
        self.channels = [32]
        for i in range(num_upsample):
            self.channels.append(64)
        # self.channels = (32, 64, 64)
        cc = self.channels[-1]

        sz = 8

        self.main = nn.Sequential()
        self.main.add_module('depth_up',
                             ConvBlock(self.in_ch, cc, kernel_size=3, pad=1, upsample=True, pad_mode=pad_mode,
                                       use_resblock=False))
        for ch in reversed(self.channels[1:-1]):
            self.main.add_module('conv_to_{}'.format(sz * 2), ConvBlock(cc, ch, kernel_size=3, pad=1, upsample=True,
                                                                        pad_mode=pad_mode, use_resblock=False))
            cc, sz = ch, sz * 2

        self.main.add_module('conv_to_{}'.format(sz * 2),
                             ConvBlock(cc, self.channels[0], kernel_size=3, pad=1,
                                       upsample=False, pad_mode=pad_mode, use_resblock=False))
        self.main.add_module('final_conv', ConvBlock(self.channels[0], num_chans, kernel_size=1, bias=True,
                                                     activation=False, batchnorm=False))
        # self.final_conv = ConvBlock(self.channels[0], num_chans, kernel_size=1, bias=True,
        #                             activation=False, batchnorm=False)
        self.decode = self.main
        # self.decode = nn.Sequential(
        #     nn.Linear(bottleneck_size, 8 * bottleneck_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(8 * bottleneck_size,
        #               num_chans * patch_size[0] * patch_size[1]),
        #     nn.Sigmoid()
        # )
        # self.decode = nn.Sequential(
        #     nn.Linear(bottleneck_size, 8 * bottleneck_size),
        #     nn.GroupNorm(8, 8 * bottleneck_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(8 * bottleneck_size,
        #               num_chans * patch_size[0] * patch_size[1]),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        conv_in = self.fc(x)
        conv_in = conv_in.view(-1, 32, 8, 8)
        out = self.decode(conv_in).view(-1, self.num_chans, *self.patch_size)
        out = torch.sigmoid(out)
        return out


# torch geometric
class PointNetPPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, axis_dim=2):
        # Message passing with "max" aggregation.
        super(PointNetPPLayer, self).__init__('max')
        self.axis_dim = axis_dim
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = nn.Sequential(nn.Linear(in_channels + axis_dim, out_channels),
                                 nn.BatchNorm1d(out_channels),
                                 nn.ReLU(),
                                 nn.Linear(out_channels, out_channels),
                                 nn.BatchNorm1d(out_channels))

    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input_pos = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input_pos = torch.cat([h_j, input_pos], dim=-1)

        return self.mlp(input_pos)  # Apply our final MLP.


class PointNetPPVariational(nn.Module):
    def __init__(self, axis_dim=2, features_dim=2, with_fps=False, zdim=128):
        super(PointNetPPVariational, self).__init__()
        self.with_fps = with_fps
        self.zdim = zdim
        self.axis_dim = axis_dim  # mu or z
        self.features_dim = features_dim  # logvar
        self.conv1 = PointNetPPLayer(self.axis_dim + self.features_dim, 64, axis_dim=axis_dim)
        self.conv2 = PointNetPPLayer(64, 128, axis_dim=axis_dim)
        self.conv3 = PointNetPPLayer(128, 256, axis_dim=axis_dim)
        self.conv4 = PointNetPPLayer(256, 512, axis_dim=axis_dim)
        self.fc = nn.Sequential(nn.Linear(512, 256, bias=True),
                                nn.ReLU(True),
                                nn.Linear(256, 128),
                                nn.ReLU(True),
                                nn.Linear(128, self.zdim * 2))

    def forward(self, features, pos, batch):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=10, batch=batch, loop=True)

        # 3. Start bipartite message passing.
        h = self.conv1(h=features, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv1 h: {h.shape}')
        if self.with_fps:
            index = fps(pos, batch=batch, ratio=0.5)
            pos = pos[index]
            h = h[index]
            batch = batch[index]
            edge_index = knn_graph(pos, k=5, batch=batch, loop=True)
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv2 h: {h.shape}')
        if self.with_fps:
            index = fps(pos, batch=batch, ratio=0.5)
            pos = pos[index]
            h = h[index]
            batch = batch[index]
            edge_index = knn_graph(pos, k=3, batch=batch, loop=True)
        h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv3 h: {h.shape}')
        #         if self.with_fps:
        #             index = fps(pos, batch=batch, ratio=0.5)
        #             pos = pos[index]
        #             h = h[index]
        #             batch = batch[index]
        #             edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        h = self.conv4(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        # print(f'conv4 h: {h.shape}')
        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        # print(f'maxpool h: {h.shape}')
        # 5. FC
        h = self.fc(h)
        mu, logvar = torch.chunk(h, chunks=2, dim=-1)
        return mu, logvar


class PointNetPPFC(nn.Module):
    def __init__(self, axis_dim=2, features_dim=2, with_fps=False, zdim=128):
        super(PointNetPPFC, self).__init__()
        self.with_fps = with_fps
        self.zdim = zdim
        self.axis_dim = axis_dim  # mu or z
        self.features_dim = features_dim  # logvar
        self.conv1 = PointNetPPLayer(self.axis_dim + self.features_dim, 64, axis_dim=axis_dim)
        self.conv2 = PointNetPPLayer(64, 128, axis_dim=axis_dim)
        self.conv3 = PointNetPPLayer(128, 256, axis_dim=axis_dim)
        self.conv4 = PointNetPPLayer(256, 512, axis_dim=axis_dim)
        # self.fc = nn.Sequential(nn.Linear(512, 256, bias=True),
        #                         nn.ReLU(True),
        #                         nn.Linear(256, 128),
        #                         nn.ReLU(True),
        #                         nn.Linear(128, self.zdim))
        self.fc = nn.Sequential(nn.Linear(512, 256, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, self.zdim))

    def forward(self, position, features):
        # position [batch_size, n_kp, 2]
        # features [batch_size, n_kp, features_dim]
        pos = position
        batch = torch.arange(pos.shape[0]).view(-1, 1).repeat(1, pos.shape[1]).view(-1).to(pos.device)
        pos = pos.view(-1, pos.shape[-1])  # [batch_size * n_kp, 2]
        features = features.view(-1, features.shape[-1])  # [batch_size * n_kp, features]
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=10, batch=batch, loop=True)

        # 3. Start bipartite message passing.
        h = self.conv1(h=features, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv1 h: {h.shape}')
        if self.with_fps:
            index = fps(pos, batch=batch, ratio=0.5)
            pos = pos[index]
            h = h[index]
            batch = batch[index]
            edge_index = knn_graph(pos, k=5, batch=batch, loop=True)
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv2 h: {h.shape}')
        if self.with_fps:
            index = fps(pos, batch=batch, ratio=0.5)
            pos = pos[index]
            h = h[index]
            batch = batch[index]
            edge_index = knn_graph(pos, k=3, batch=batch, loop=True)
        h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv3 h: {h.shape}')
        #         if self.with_fps:
        #             index = fps(pos, batch=batch, ratio=0.5)
        #             pos = pos[index]
        #             h = h[index]
        #             batch = batch[index]
        #             edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        h = self.conv4(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        # print(f'conv4 h: {h.shape}')
        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        # print(f'maxpool h: {h.shape}')
        # 5. FC
        h = self.fc(h)
        return h


class PointNetPPFCLocal(nn.Module):
    def __init__(self, axis_dim=2, features_dim=2, with_fps=False, zdim=128):
        super(PointNetPPFCLocal, self).__init__()
        self.with_fps = with_fps
        self.zdim = zdim
        self.axis_dim = axis_dim  # mu or z
        self.features_dim = features_dim  # logvar
        self.conv1 = PointNetPPLayer(self.axis_dim + self.features_dim, 64, axis_dim=axis_dim)
        self.conv2 = PointNetPPLayer(64, 128, axis_dim=axis_dim)
        self.conv3 = PointNetPPLayer(128, 256, axis_dim=axis_dim)
        self.conv4 = PointNetPPLayer(256, 512, axis_dim=axis_dim)
        # self.fc = nn.Sequential(nn.Linear(512, 256, bias=True),
        #                         nn.ReLU(True),
        #                         nn.Linear(256, 128),
        #                         nn.ReLU(True),
        #                         nn.Linear(128, self.zdim))
        self.fc = nn.Sequential(nn.Linear(512, 256, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, self.zdim))

    def forward(self, position, features):
        # position [batch_size, n_kp, 2]
        # features [batch_size, n_kp, features_dim]
        pos = position
        batch = torch.arange(pos.shape[0]).view(-1, 1).repeat(1, pos.shape[1]).view(-1).to(pos.device)
        pos = pos.view(-1, pos.shape[-1])  # [batch_size * n_kp, 2]
        features = features.view(-1, features.shape[-1])  # [batch_size * n_kp, features]
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=10, batch=batch, loop=True)

        # 3. Start bipartite message passing.
        h = self.conv1(h=features, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv1 h: {h.shape}')
        if self.with_fps:
            index = fps(pos, batch=batch, ratio=0.5)
            pos = pos[index]
            h = h[index]
            batch = batch[index]
            edge_index = knn_graph(pos, k=5, batch=batch, loop=True)
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv2 h: {h.shape}')
        if self.with_fps:
            index = fps(pos, batch=batch, ratio=0.5)
            pos = pos[index]
            h = h[index]
            batch = batch[index]
            edge_index = knn_graph(pos, k=3, batch=batch, loop=True)
        h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv3 h: {h.shape}')
        #         if self.with_fps:
        #             index = fps(pos, batch=batch, ratio=0.5)
        #             pos = pos[index]
        #             h = h[index]
        #             batch = batch[index]
        #             edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        h = self.conv4(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        # print(f'conv4 h: {h.shape}')
        # 4. Global Pooling.
        # h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        # print(f'maxpool h: {h.shape}')
        # 5. FC
        h = self.fc(h)
        return h


class PointNetPPToCNN(nn.Module):
    def __init__(self, axis_dim=2, target_hw=16, n_kp=8, pad_mode='reflect', with_fps=False, features_dim=2):
        super(PointNetPPToCNN, self).__init__()
        # features_dim : 2 [logvar] + additional features
        self.with_fps = with_fps
        self.axis_dim = axis_dim  # mu
        self.features_dim = features_dim  # logvar, features
        self.conv1 = PointNetPPLayer(self.axis_dim + self.features_dim, 64, axis_dim=axis_dim)
        self.conv2 = PointNetPPLayer(64, 128, axis_dim=axis_dim)
        self.conv3 = PointNetPPLayer(128, 256, axis_dim=axis_dim)
        self.conv4 = PointNetPPLayer(256, 512, axis_dim=axis_dim)

        self.n_kp = n_kp
        # self.pointnet = nn.Sequential(
        #     nn.Conv1d(in_channels=axis_dim, out_channels=64, kernel_size=1,
        #               bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(64),
        #
        #     nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
        #               bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(128),
        #
        #     nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
        #               bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(256),
        #
        #     nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
        #               bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(512),
        # )

        fc_out_dim = self.n_kp * 8 * 8
        # self.fc = nn.Sequential(nn.Linear(512, 256, bias=True),
        #                         nn.ReLU(True),
        #                         nn.Linear(256, 128),
        #                         nn.ReLU(True),
        #                         nn.Linear(128, fc_out_dim),
        #                         nn.ReLU(True))

        self.fc = nn.Sequential(nn.Linear(512, fc_out_dim, bias=True),
                                nn.ReLU(True))

        num_upsample = int(np.log(target_hw) // np.log(2)) - 3
        print(f'pointnet to cnn num upsample: {num_upsample}')
        self.cnn = nn.Sequential()
        for i in range(num_upsample):
            self.cnn.add_module(f'depth_up_{i}', ConvBlock(n_kp, n_kp, kernel_size=3, pad=1,
                                                           upsample=True, pad_mode=pad_mode))

    def forward(self, position, features):
        # position [batch_size, n_kp, 2]
        # features [batch_size, n_kp, features_dim]
        pos = position
        batch = torch.arange(pos.shape[0]).view(-1, 1).repeat(1, pos.shape[1]).view(-1).to(pos.device)
        pos = pos.view(-1, pos.shape[-1])  # [batch_size * n_kp, 2]
        features = features.view(-1, features.shape[-1])  # [batch_size * n_kp, features]
        # x [batch_size, n_kp, 2 or features_dim]
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=10, batch=batch, loop=True)

        # 3. Start bipartite message passing.
        h = self.conv1(h=features, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv1 h: {h.shape}')
        if self.with_fps:
            index = fps(pos, batch=batch, ratio=0.5)
            pos = pos[index]
            h = h[index]
            batch = batch[index]
            edge_index = knn_graph(pos, k=5, batch=batch, loop=True)
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv2 h: {h.shape}')
        if self.with_fps:
            index = fps(pos, batch=batch, ratio=0.5)
            pos = pos[index]
            h = h[index]
            batch = batch[index]
            edge_index = knn_graph(pos, k=3, batch=batch, loop=True)
        h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        #         print(f'conv3 h: {h.shape}')
        #         if self.with_fps:
        #             index = fps(pos, batch=batch, ratio=0.5)
        #             pos = pos[index]
        #             h = h[index]
        #             batch = batch[index]
        #             edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        h = self.conv4(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        # print(f'conv4 h: {h.shape}')
        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        # print(f'maxpool h: {h.shape}')
        # 5. FC
        h = self.fc(h)
        h = h.view(-1, self.n_kp, 8, 8)  # [batch_size, n_kp, 4, 4]
        cnn_out = self.cnn(h)  # [batch_size, n_kp, target_hw, target_hw]

        # x_in = x.transpose(2, 1)  # [batch_size, 2 or features_dim, n_kp]
        # output = self.pointnet(x_in)
        # output2 = output.max(dim=2)[0]
        # lin_out = self.fc(output2)  # [batch_size, n_kp * 4 * 4]
        # lin_out = lin_out.view(-1, self.n_kp, 4, 4)  # [batch_size, n_kp, 4, 4]
        # cnn_out = self.cnn(lin_out)  # [batch_size, n_kp, target_hw, target_hw]
        return cnn_out


class GCNDelta(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, graph_type='radius', axis_dim=2, radius=0.2,
                 k_nn=10, max_delta=0.4, use_gated_gcn=False, horizon=1):
        super().__init__()
        assert graph_type in ["radius", "knn"]
        self.graph_type = graph_type
        self.radius = radius
        self.k_nn = k_nn
        self.hidden_channels = hidden_channels
        self.num_features = num_features
        self.axis_dim = axis_dim
        self.max_delta = max_delta
        self.use_gated_gcn = use_gated_gcn
        self.horizon = horizon  # should be converted to one-hot encoding

        # TODO: dropout? embedding?

        horizon_dim = self.horizon if self.horizon > 1 else 0
        in_features_dim = self.axis_dim + self.num_features + horizon_dim
        out_features_dim = self.axis_dim + self.num_features
        if self.use_gated_gcn:
            self.conv1 = ResGatedGraphConv(in_features_dim, hidden_channels)
            self.conv2 = ResGatedGraphConv(hidden_channels, out_features_dim)
        else:
            self.conv1 = GCNConv(in_features_dim, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_features_dim)
        self.conv_in_between = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                             nn.ReLU(),
                                             nn.Linear(hidden_channels, hidden_channels))

    def forward(self, pos, features):
        # pos: [bs, n_kp, 2]
        # features: [bs, n_kp, feat_dim]
        batch_size, n_kp = pos.shape[0], pos.shape[1]
        batch = torch.arange(pos.shape[0]).view(-1, 1).repeat(1, pos.shape[1]).view(-1).to(pos.device)
        pos = pos.view(-1, pos.shape[-1])  # [batch_size * n_kp, 2]
        features = features.view(-1, features.shape[-1])  # [batch_size * n_kp, features]
        if self.graph_type == 'radius':
            edge_index = radius_graph(pos, r=self.radius, batch=batch, loop=True, max_num_neighbors=self.k_nn)
        elif self.graph_type == 'knn':
            edge_index = knn_graph_nn(pos, k=self.k_nn, batch=batch, loop=True)
        else:
            raise NotImplementedError('graph type unrecognized')

        x_in = torch.cat([pos, features], dim=-1)
        x_in = self.conv1(x_in, edge_index)
        x_in = torch.relu(x_in)
        x_in = self.conv_in_between(x_in)
        #         x = F.dropout(x, p=0.5, training=self.training)
        x_in = self.conv2(x_in, edge_index)

        delta_pos, delta_features = torch.split(x_in, [self.axis_dim, self.num_features], dim=-1)
        delta_pos = self.max_delta * torch.tanh(delta_pos)
        delta_features = 0.1 * self.max_delta * torch.tanh(delta_features)
        # delta_features = torch.tanh(delta_features)
        # delta_features = self.max_delta * torch.tanh(delta_features)

        delta_pos = delta_pos.view(batch_size, n_kp, -1)
        delta_features = delta_features.view(batch_size, n_kp, -1)
        return delta_pos, delta_features
