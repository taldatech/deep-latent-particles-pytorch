"""
Interactive demo to observe and explore the learned particles.
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from models import KeyPointVAE
from utils.util_func import reparameterize

import matplotlib
from matplotlib.widgets import Slider, Button
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import argparse

matplotlib.use('Qt5Agg')


def update_from_slider(val):
    for i in np.arange(N):
        yvals[i] = sliders_y[i].val
        # xvals[i] = sliders_x[i].val
        if learned_feature_dim > 0:
            feature_1_vals[i] = sliders_features[i].val
    update(val)


def update(val):
    global yvals
    global xvals
    global dec_bone
    global learned_feature_dim
    if learned_feature_dim > 0:
        global feature_1_vals
    # update curve
    for i in np.arange(N):
        if learned_feature_dim > 0:
            # print(f'{i}: {feature_1_vals[i]}')
            feature_1_vals[i] = sliders_features[i].val
            # print(f'{i}: {feature_1_vals[i]}')
    l.set_offsets(np.c_[xvals, yvals])
    # convert to tensors
    new_mu = torch.from_numpy(np.stack([yvals, xvals], axis=-1)).unsqueeze(0).to(device) / (image_size - 1)  # [0, 1]
    new_mu = new_mu * (kp_range[1] - kp_range[0]) + kp_range[0]  # [kp_range[0], kp_range[1]]
    delta_mu = new_mu - original_mu
    # print(f'delta_mu: {delta_mu}')
    if learned_feature_dim > 0:
        new_features = torch.from_numpy(feature_1_vals[None, :, None]).to(device)
        new_features = torch.cat([mu_features[:, :, :-1], new_features], dim=-1)
    else:
        new_features = None
    with torch.no_grad():
        rec_new, _, _ = model.decode_all(new_mu, new_features, kp_heatmap, obj_on, deterministic=deterministic,
                                         order_weights=order_weights)
        rec_new = rec_new.clamp(0, 1)

    image_rec_new = rec_new[0].permute(1, 2, 0).data.cpu().numpy()
    m.set_data(image_rec_new)
    # redraw canvas while idle
    fig.canvas.draw_idle()


def reset(event):
    global yvals
    global xvals
    global learned_feature_dim
    if learned_feature_dim > 0:
        global feature_1_vals
    # reset the values
    xvals = mu[0, :, 1].data.cpu().numpy() * (image_size - 1)
    yvals = mu[0, :, 0].data.cpu().numpy() * (image_size - 1)
    if learned_feature_dim > 0:
        # a slider for the last feature dimension
        # feature_1_vals = mu_features[0, :, 0].data.cpu().numpy()
        feature_1_vals = mu_features[0, :, -1].data.cpu().numpy()
    for i in np.arange(N):
        sliders_y[i].reset()
        if learned_feature_dim > 0:
            sliders_features[i].reset()
    l.set_offsets(np.c_[xvals, yvals])
    m.set_data(image_rec)
    # redraw canvas while idle
    fig.canvas.draw_idle()


def button_press_callback(event):
    'whenever a mouse button is pressed'
    global pind
    if event.inaxes is None:
        return
    if event.button != 1:
        return
    pind = get_ind_under_point(event)


def button_release_callback(event):
    'whenever a mouse button is released'
    global pind
    if event.button != 1:
        return
    pind = None


def get_ind_under_point(event):
    'get the index of the vertex under point if within epsilon tolerance'
    t = ax1.transData.inverted()
    tinv = ax1.transData
    xy = t.transform([event.x, event.y])
    xr = np.reshape(xvals, (np.shape(xvals)[0], 1))
    yr = np.reshape(yvals, (np.shape(yvals)[0], 1))
    xy_vals = np.append(xr, yr, 1)
    xyt = tinv.transform(xy_vals)
    xt, yt = xyt[:, 0], xyt[:, 1]
    d = np.hypot(xt - event.x, yt - event.y)
    indseq, = np.nonzero(d == d.min())
    ind = indseq[0]

    if d[ind] >= epsilon:
        ind = None
    return ind


def motion_notify_callback(event):
    'on mouse movement'
    global xvals
    global yvals
    if pind is None:
        return
    if event.inaxes is None:
        return
    if event.button != 1:
        return

    # update yvals
    # print('motion x: {0}; y: {1}'.format(event.xdata,event.ydata))
    # print(f'delta: x: {event.xdata - xvals[pind]}. y: {event.ydata - yvals[pind]}')
    delta_x = event.xdata - xvals[pind]
    delta_y = event.ydata - yvals[pind]
    yvals[pind] = event.ydata
    xvals[pind] = event.xdata

    # yvals[pind + 1] = yvals[pind + 1] + delta_y
    # xvals[pind + 1] = xvals[pind + 1] + delta_x

    update(None)

    # update curve via sliders and draw
    sliders_y[pind].set_val(yvals[pind])
    # sliders_y[pind + 1].set_val(yvals[pind + 1])

    # sliders_x[pind].set_val(xvals[pind])
    fig.canvas.draw_idle()


def bn_eval(model):
    """
    https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/67
    for batch_size = 1  don't use the running stats
    """
    for m in model.modules():
        for child in m.children():
            if type(child) == torch.nn.BatchNorm2d or type(child) == torch.nn.BatchNorm1d:
                child.track_running_stats = False
                child.running_mean = None
                child.running_var = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DLP Interactive Demo")
    parser.add_argument("-d", "--dataset", type=str, default='celeb',
                        help="dataset of pretrained model: ['celeb', 'traffic', 'clevrer']")
    parser.add_argument("-i", "--index", type=int,
                        help="index of image in ./checkpoints/sample_images/dataset/", default=0)
    args = parser.parse_args()
    # hyper-parameters for model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_logsoftmax = False
    pad_mode = 'replicate'
    sigma = 0.1  # default sigma for the gaussian maps
    dropout = 0.0
    n_kp = 1  # num kp per patch
    kp_range = (-1, 1)
    kp_activation = "tanh"
    mask_threshold = 0.2
    learn_order = False

    # ds = 'celeb'
    # ds = 'traffic'
    # ds = 'clevrer'
    ds = args.dataset

    # image_idx = 2
    image_idx = args.index
    image_idx = max(0, image_idx)

    if ds == 'celeb':
        path_to_model_ckpt = './checkpoints/dlp_celeba_gauss_pointnetpp_feat.pth'
        image_size = 128
        ch = 3
        enc_channels = [32, 64, 128, 256]
        prior_channels = (16, 32, 64)
        imwidth = 160
        crop = 16
        n_kp_enc = 30  # total kp to output from the encoder / filter from prior
        n_kp_prior = 50  # total kp to filter from prior
        use_object_enc = True
        use_object_dec = False
        learned_feature_dim = 10
        patch_size = 8
        anchor_s = 0.125
        dec_bone = "gauss_pointnetpp_feat"
    elif ds == 'traffic':
        path_to_model_ckpt = './checkpoints/dlp_traffic_gauss_pointnetpp.pth'
        image_size = 128
        ch = 3
        enc_channels = [32, 64, 128, 256]
        prior_channels = (16, 32, 64)
        imwidth = 160
        crop = 16
        n_kp_enc = 15  # total kp to output from the encoder / filter from prior
        n_kp_prior = 20  # total kp to filter from prior
        use_object_enc = True
        use_object_dec = True
        learned_feature_dim = 20
        patch_size = 16
        anchor_s = 0.25
        dec_bone = "gauss_pointnetpp"
    elif ds == 'clevrer':
        path_to_model_ckpt = './checkpoints/dlp_clevrer_gauss_pointnetpp.pth'
        image_size = 128
        ch = 3
        enc_channels = [32, 64, 128, 256]
        prior_channels = (16, 32, 64)
        imwidth = 160
        crop = 16
        n_kp_enc = 10  # total kp to output from the encoder / filter from prior
        n_kp_prior = 20  # total kp to filter from prior
        use_object_enc = True
        use_object_dec = True
        learned_feature_dim = 5
        patch_size = 16
        anchor_s = 0.25
        dec_bone = "gauss_pointnetpp"
    else:
        raise NotImplementedError

    model = KeyPointVAE(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                        image_size=image_size, n_kp=n_kp, learned_feature_dim=learned_feature_dim,
                        use_logsoftmax=use_logsoftmax, pad_mode=pad_mode, sigma=sigma,
                        dropout=dropout, dec_bone=dec_bone, patch_size=patch_size, n_kp_enc=n_kp_enc,
                        n_kp_prior=n_kp_prior, kp_range=kp_range, kp_activation=kp_activation,
                        mask_threshold=mask_threshold, use_object_enc=use_object_enc,
                        use_object_dec=use_object_dec, anchor_s=anchor_s, learn_order=learn_order).to(device)
    model.load_state_dict(torch.load(path_to_model_ckpt, map_location=device), strict=False)
    model.eval()
    print("loaded model from checkpoint")
    if ds == 'celeb':
        # load image
        path_to_images = ['./checkpoints/sample_images/celeb/1.jpg', './checkpoints/sample_images/celeb/2.jpg',
                          './checkpoints/sample_images/celeb/3.jpg', './checkpoints/sample_images/celeb/4.jpg',
                          './checkpoints/sample_images/celeb/5.jpg', './checkpoints/sample_images/celeb/6.jpg',
                          './checkpoints/sample_images/celeb/7.jpg', './checkpoints/sample_images/celeb/8.jpg']
        image_idx = min(image_idx, len(path_to_images) - 1)
        path_to_image = path_to_images[image_idx]
        im = Image.open(path_to_image)
        # move head up a bit
        vertical_shift = 30
        initial_crop = lambda im: transforms.functional.crop(im, 30, 0, 178, 178)
        initial_transforms = transforms.Compose([initial_crop, transforms.Resize(imwidth)])
        trans = transforms.ToTensor()
        data = trans(initial_transforms(im.convert("RGB")))
        if crop != 0:
            data = data[:, crop:-crop, crop:-crop]
        data = data.unsqueeze(0).to(device)
    elif ds == 'traffic':
        path_to_images = ['./checkpoints/sample_images/traffic/1.png',]
        image_idx = min(image_idx, len(path_to_images) - 1)
        path_to_image = path_to_images[image_idx]
        im = Image.open(path_to_image)
        im = im.convert('RGB')
        im = im.crop((60, 0, 480, 420))
        im = im.resize((image_size, image_size), Image.BICUBIC)
        trans = transforms.ToTensor()
        data = trans(im)
        data = data.unsqueeze(0).to(device)
        x = data
    elif ds == 'clevrer':
        path_to_images = ['./checkpoints/sample_images/clevrer/1.png',]
        image_idx = min(image_idx, len(path_to_images) - 1)
        path_to_image = path_to_images[image_idx]
        im = Image.open(path_to_image)
        im = im.convert('RGB')
        im = im.resize((image_size, image_size), Image.BICUBIC)
        trans = transforms.ToTensor()
        data = trans(im)
        data = data.unsqueeze(0).to(device)
        x = data
    else:
        raise NotImplementedError

    with torch.no_grad():
        deterministic = True
        enc_out = model.encode_all(data, return_heatmap=True, deterministic=deterministic)
        mu, logvar, kp_heatmap, mu_features, logvar_features, obj_on, order_weights = enc_out
        if deterministic:
            z = mu
            z_features = mu_features
        else:
            z = reparameterize(mu, logvar)
            z_features = reparameterize(mu_features, logvar_features)

        if learn_order:
            order_of_kp = [torch.argmax(order_weights[0][i]).item() for i in range(order_weights.shape[-1])]
            print(f'order of kp: {order_of_kp}')
        if obj_on is not None:
            print(f'obj_on: {obj_on[0].data.cpu()}')

        rec, _, _ = model.decode_all(z, z_features, kp_heatmap, obj_on, deterministic=deterministic,
                                     order_weights=order_weights)
        rec = rec.clamp(0, 1)

    # top-k
    with torch.no_grad():
        logvar_sum = logvar.sum(-1)
        logvar_topk = torch.topk(logvar_sum, k=5, dim=-1, largest=False)
        indices = logvar_topk[1]  # [batch_size, topk]
        batch_indices = torch.arange(mu.shape[0]).view(-1, 1).to(mu.device)
        topk_kp = mu[batch_indices, indices]

    N = mu.shape[1]
    xmin = 0
    xmax = image_size

    x = np.linspace(xmin, xmax, N)

    mu = mu.clamp(kp_range[0], kp_range[1])
    original_mu = mu.clone()
    mu = (mu - kp_range[0]) / (kp_range[1] - kp_range[0])
    xvals = mu[0, :, 1].data.cpu().numpy() * (image_size - 1)
    yvals = mu[0, :, 0].data.cpu().numpy() * (image_size - 1)
    if learned_feature_dim > 0:
        # feature_1_vals = mu_features[0, :, 0].data.cpu().numpy()
        feature_1_vals = mu_features[0, :, -1].data.cpu().numpy()

    # set up a plot for topk
    topk_kp = topk_kp.clamp(kp_range[0], kp_range[1])
    topk_kp = (topk_kp - kp_range[0]) / (kp_range[1] - kp_range[0])
    xvals_topk = topk_kp[0, :, 1].data.cpu().numpy() * (image_size - 1)
    yvals_topk = topk_kp[0, :, 0].data.cpu().numpy() * (image_size - 1)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    image = data[0].permute(1, 2, 0).data.cpu().numpy()
    ax1.imshow(image)
    ax1.scatter(xvals, yvals, label='original', s=70)
    ax1.set_axis_off()
    ax1.set_title('all particles')
    fig = plt.figure(figsize=(10, 10))
    ax2 = fig.add_subplot(111)
    ax2.imshow(image)
    ax2.scatter(xvals_topk, yvals_topk, label='topk', s=70, color='red')
    ax2.set_axis_off()
    ax2.set_title('top-5 lowest variance particles')

    # figure.subplot.right
    mpl.rcParams['figure.subplot.right'] = 0.8

    # set up a plot
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 15.0), sharex=True)
    ax1, ax2 = axes

    image = data[0].permute(1, 2, 0).data.cpu().numpy()
    ax1.imshow(image)
    ax1.set_axis_off()
    ax2.set_axis_off()
    image_rec = rec[0].permute(1, 2, 0).data.cpu().numpy()
    m = ax2.imshow(image_rec)

    pind = None  # active point
    epsilon = 10  # max pixel distance

    ax1.scatter(xvals, yvals, label='original', s=70)
    l = ax1.scatter(xvals, yvals, color='red', marker='*', s=70)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    sliders_y = []
    sliders_x = []
    if learned_feature_dim > 0:
        sliders_features = []

    for i in np.arange(N):
        slider_width = 0.04 if learned_feature_dim > 0 else 0.12
        axamp = plt.axes([0.84, 0.85 - (i * 0.025), slider_width, 0.01])
        # Slider y
        s_y = Slider(axamp, 'p_y{0}'.format(i), 0, image_size, valinit=yvals[i])
        sliders_y.append(s_y)
        if learned_feature_dim > 0:
            axamp_f = plt.axes([0.93, 0.85 - (i * 0.025), slider_width, 0.01])
            s_feat = Slider(axamp_f, f'f_1', -5, 5, valinit=feature_1_vals[i])
            sliders_features.append(s_feat)

    for i in np.arange(N):
        sliders_y[i].on_changed(update_from_slider)
        if learned_feature_dim > 0:
            sliders_features[i].on_changed(update_from_slider)

    axres = plt.axes([0.84, 0.85 - ((N) * 0.025), 0.12, 0.01])
    bres = Button(axres, 'Reset')
    bres.on_clicked(reset)

    fig.canvas.mpl_connect('button_press_event', button_press_callback)
    fig.canvas.mpl_connect('button_release_event', button_release_callback)
    fig.canvas.mpl_connect('motion_notify_event', motion_notify_callback)

    plt.show()
