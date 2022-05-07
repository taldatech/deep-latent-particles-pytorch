"""
Loss functions implementations used in the optimization of DLP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# functions
def batch_pairwise_kl(mu_x, logvar_x, mu_y, logvar_y, reverse_kl=False):
    """
    Calculate batch-wise KL-divergence
    mu_x, logvar_x: [batch_size, n_x, points_dim]
    mu_y, logvar_y: [batch_size, n_y, points_dim]
    kl = -0.5 * Σ_points_dim (1 + logvar_x - logvar_y - exp(logvar_x)/exp(logvar_y)
                    - ((mu_x - mu_y) ** 2)/exp(logvar_y))
    """
    if reverse_kl:
        mu_a, logvar_a = mu_y, logvar_y
        mu_b, logvar_b = mu_x, logvar_x
    else:
        mu_a, logvar_a = mu_x, logvar_x
        mu_b, logvar_b = mu_y, logvar_y
    bs, n_a, points_dim = mu_a.size()
    _, n_b, _ = mu_b.size()
    logvar_aa = logvar_a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    logvar_bb = logvar_b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    mu_aa = mu_a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    mu_bb = mu_b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    p_kl = -0.5 * (1 + logvar_aa - logvar_bb - logvar_aa.exp() / logvar_bb.exp()
                   - ((mu_aa - mu_bb) ** 2) / logvar_bb.exp()).sum(-1)  # [batch_size, n_x, n_y]
    return p_kl


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


# classes
class ChamferLossKL(nn.Module):
    """
    Calculates the KL-divergence between two sets of (R.V.) particle coordinates.
    """
    def __init__(self, use_reverse_kl=False):
        super(ChamferLossKL, self).__init__()
        self.use_reverse_kl = use_reverse_kl

    def forward(self, mu_preds, logvar_preds, mu_gts, logvar_gts):
        p_kl = batch_pairwise_kl(mu_preds, logvar_preds, mu_gts, logvar_gts, reverse_kl=False)
        if self.use_reverse_kl:
            p_rkl = batch_pairwise_kl(mu_preds, logvar_preds, mu_gts, logvar_gts, reverse_kl=True)
            p_kl = 0.5 * (p_kl + p_rkl.transpose(2, 1))
        mins, _ = torch.min(p_kl, 1)
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(p_kl, 2)
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2


class NetVGGFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = models.vgg16(pretrained=True)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGDistance(nn.Module):

    def __init__(self, layer_ids=(2, 7, 12, 21, 30), accumulate_mode='sum', device=torch.device("cpu")):
        super().__init__()

        self.vgg = NetVGGFeatures(layer_ids).to(device)
        self.layer_ids = layer_ids
        self.accumulate_mode = accumulate_mode
        self.device = device

    def forward(self, I1, I2, reduction='sum', only_image=False):
        b_sz = I1.size(0)
        num_ch = I1.size(1)

        if self.accumulate_mode == 'sum':
            loss = ((I1 - I2) ** 2).view(b_sz, -1).sum(1)
        else:
            loss = ((I1 - I2) ** 2).view(b_sz, -1).mean(1)

        if num_ch == 1:
            I1 = I1.repeat(1, 3, 1, 1)
            I2 = I2.repeat(1, 3, 1, 1)
        f1 = self.vgg(I1)
        f2 = self.vgg(I2)

        if not only_image:
            for i in range(len(self.layer_ids)):
                if self.accumulate_mode == 'sum':
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, -1).sum(1)
                else:
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, -1).mean(1)
                loss = loss + layer_loss

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_dimensions(self, device=torch.device("cpu")):
        dims = []
        dummy_input = torch.zeros(1, 3, 128, 128).to(device)
        dims.append(dummy_input.view(1, -1).size(1))
        f = self.vgg(dummy_input)
        for i in range(len(self.layer_ids)):
            dims.append(f[i].view(1, -1).size(1))
        return dims


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P
