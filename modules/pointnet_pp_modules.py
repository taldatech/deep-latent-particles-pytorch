# imports
import numpy as np
# torch and friends
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_max_pool
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
                self.main.add_module('reflect_pad', nn.ReplicationPad2d(1))
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
        if upsample:
            self.main.add_module(f'upsample_bilinear_2',
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self, x):
        y = self.main(x)
        return y


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
                                nn.SiLU(inplace=True),
                                nn.Linear(256, 128),
                                nn.SiLU(inplace=True),
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


class PointNetPPNoPool(nn.Module):
    def __init__(self, axis_dim=2, target_hw=16, n_kp=8, pad_mode='reflect', with_fps=False, features_dim=2):
        super(PointNetPPNoPool, self).__init__()
        # features_dim : 2 [logvar] + additional features
        self.with_fps = with_fps
        self.axis_dim = axis_dim  # mu
        self.features_dim = features_dim  # logvar, features
        self.target_hw = target_hw
        self.n_kp = n_kp
        self.conv1 = PointNetPPLayer(self.axis_dim + self.features_dim, 256, axis_dim=axis_dim)
        self.conv2 = PointNetPPLayer(256, 256, axis_dim=axis_dim)
        self.conv3 = PointNetPPLayer(256, 256, axis_dim=axis_dim)
        self.conv4 = PointNetPPLayer(256, target_hw ** 2, axis_dim=axis_dim)
        self.cnn = nn.Conv2d(in_channels=n_kp, out_channels=n_kp, kernel_size=1)

        # self.n_kp = n_kp
        # # self.pointnet = nn.Sequential(
        # #     nn.Conv1d(in_channels=axis_dim, out_channels=64, kernel_size=1,
        # #               bias=False),
        # #     nn.ReLU(inplace=True),
        # #     nn.BatchNorm1d(64),
        # #
        # #     nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
        # #               bias=False),
        # #     nn.ReLU(inplace=True),
        # #     nn.BatchNorm1d(128),
        # #
        # #     nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
        # #               bias=False),
        # #     nn.ReLU(inplace=True),
        # #     nn.BatchNorm1d(256),
        # #
        # #     nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
        # #               bias=False),
        # #     nn.ReLU(inplace=True),
        # #     nn.BatchNorm1d(512),
        # # )
        #
        # fc_out_dim = self.n_kp * 4 * 4
        # self.fc = nn.Sequential(nn.Linear(512, 256, bias=True),
        #                         nn.ReLU(True),
        #                         nn.Linear(256, 128),
        #                         nn.ReLU(True),
        #                         nn.Linear(128, fc_out_dim),
        #                         nn.ReLU(True))
        #
        # num_upsample = int(np.log(target_hw) // np.log(2)) - 2
        # print(f'pointnet to cnn num upsample: {num_upsample}')
        # self.cnn = nn.Sequential()
        # for i in range(num_upsample):
        #     self.cnn.add_module(f'depth_up_{i}', ConvBlock(n_kp, n_kp, kernel_size=3, pad=1,
        #                                                    upsample=True, pad_mode=pad_mode))

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
        h = h.view(-1, self.n_kp, self.target_hw, self.target_hw)
        cnn_out = self.cnn(h)
        # print(f'conv4 h: {h.shape}')
        # 4. Global Pooling.
        # h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        # print(f'maxpool h: {h.shape}')
        # 5. FC
        # h = self.fc(h)
        # h = h.view(-1, self.n_kp, 4, 4)  # [batch_size, n_kp, 4, 4]
        # cnn_out = self.cnn(h)  # [batch_size, n_kp, target_hw, target_hw]

        # x_in = x.transpose(2, 1)  # [batch_size, 2 or features_dim, n_kp]
        # output = self.pointnet(x_in)
        # output2 = output.max(dim=2)[0]
        # lin_out = self.fc(output2)  # [batch_size, n_kp * 4 * 4]
        # lin_out = lin_out.view(-1, self.n_kp, 4, 4)  # [batch_size, n_kp, 4, 4]
        # cnn_out = self.cnn(lin_out)  # [batch_size, n_kp, target_hw, target_hw]
        return cnn_out


class PointNetPPOnePool(nn.Module):
    def __init__(self, axis_dim=2, target_hw=16, n_kp=8, pad_mode='reflect', with_fps=False, features_dim=2):
        super(PointNetPPOnePool, self).__init__()
        # features_dim : 2 [logvar] + additional features
        self.with_fps = with_fps
        self.axis_dim = axis_dim  # mu
        self.features_dim = features_dim  # logvar, features
        self.target_hw = target_hw
        self.n_kp = n_kp
        self.conv1 = PointNetPPLayer(self.axis_dim + self.features_dim, 256, axis_dim=axis_dim)
        self.conv2 = PointNetPPLayer(256, 256, axis_dim=axis_dim)
        self.conv3 = PointNetPPLayer(256, 256, axis_dim=axis_dim)
        self.conv4 = PointNetPPLayer(256, target_hw ** 2, axis_dim=axis_dim)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=n_kp, kernel_size=1)

        # self.n_kp = n_kp
        # # self.pointnet = nn.Sequential(
        # #     nn.Conv1d(in_channels=axis_dim, out_channels=64, kernel_size=1,
        # #               bias=False),
        # #     nn.ReLU(inplace=True),
        # #     nn.BatchNorm1d(64),
        # #
        # #     nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
        # #               bias=False),
        # #     nn.ReLU(inplace=True),
        # #     nn.BatchNorm1d(128),
        # #
        # #     nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
        # #               bias=False),
        # #     nn.ReLU(inplace=True),
        # #     nn.BatchNorm1d(256),
        # #
        # #     nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
        # #               bias=False),
        # #     nn.ReLU(inplace=True),
        # #     nn.BatchNorm1d(512),
        # # )
        #
        # fc_out_dim = self.n_kp * 4 * 4
        # self.fc = nn.Sequential(nn.Linear(512, 256, bias=True),
        #                         nn.ReLU(True),
        #                         nn.Linear(256, 128),
        #                         nn.ReLU(True),
        #                         nn.Linear(128, fc_out_dim),
        #                         nn.ReLU(True))
        #
        # num_upsample = int(np.log(target_hw) // np.log(2)) - 2
        # print(f'pointnet to cnn num upsample: {num_upsample}')
        # self.cnn = nn.Sequential()
        # for i in range(num_upsample):
        #     self.cnn.add_module(f'depth_up_{i}', ConvBlock(n_kp, n_kp, kernel_size=3, pad=1,
        #                                                    upsample=True, pad_mode=pad_mode))

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
        h = h.view(-1, 1, self.target_hw, self.target_hw)
        cnn_out = self.cnn(h)
        # print(f'maxpool h: {h.shape}')
        # 5. FC
        # h = self.fc(h)
        # h = h.view(-1, self.n_kp, 4, 4)  # [batch_size, n_kp, 4, 4]
        # cnn_out = self.cnn(h)  # [batch_size, n_kp, target_hw, target_hw]

        # x_in = x.transpose(2, 1)  # [batch_size, 2 or features_dim, n_kp]
        # output = self.pointnet(x_in)
        # output2 = output.max(dim=2)[0]
        # lin_out = self.fc(output2)  # [batch_size, n_kp * 4 * 4]
        # lin_out = lin_out.view(-1, self.n_kp, 4, 4)  # [batch_size, n_kp, 4, 4]
        # cnn_out = self.cnn(lin_out)  # [batch_size, n_kp, target_hw, target_hw]
        return cnn_out
