"""
functions and classes to process the CLEVRER dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils.tps as tps

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def list_images_in_dir(path):
    valid_images = [".jpg", ".gif", ".png"]
    img_list = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img_list.append(os.path.join(path, f))
    return img_list


def prepare_numpy_file(path_to_image_dir, image_size=128, frameskip=1, start_frame=1):
    # path_to_image_dir = '/media/newhd/data/traffic_data/rimon_frames/'
    img_list = list_images_in_dir(path_to_image_dir)
    img_list = sorted(img_list, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
    img_list = [img_list[i] for i in range(len(img_list)) if
                abs(int(img_list[i].split('/')[-1].split('_')[-1].split('.')[0])) % 1000 > start_frame]
    print(f'img_list: {len(img_list)}, 0: {img_list[0]}, -1: {img_list[-1]}')
    img_np_list = []
    for i in tqdm(range(len(img_list))):
        if i % frameskip != 0:
            continue
        img = Image.open(img_list[i])
        img = img.convert('RGB')
        #         img = img.crop((60, 0, 480, 420))
        img = img.resize((image_size, image_size), Image.BICUBIC)
        img_np = np.asarray(img)
        img_np_list.append(img_np)
    img_np_array = np.stack(img_np_list, axis=0)
    print(f'img_np_array: {img_np_array.shape}')
    save_path = os.path.join(path_to_image_dir, f'clevrer_img{image_size}np_fs{frameskip}.npy')
    np.save(save_path, img_np_array)
    print(f'file save at @ {save_path}')


class CLEVRERDataset(Dataset):
    def __init__(self, path_to_npy, image_size=128, transform=None, mode='single', train=True, horizon=3,
                 frames_per_video=34, video_as_index=False):
        super(CLEVRERDataset, self).__init__()
        assert mode in ['single', 'frames', 'tps', 'horizon']
        self.mode = mode
        self.frames_per_video = frames_per_video
        self.horizon = horizon if (horizon > 0 and self.mode == 'horizon') else self.frames_per_video
        self.train_mode = train
        if train:
            print(f'clevrer dataset mode: {self.mode}')
            if self.mode == 'horizon':
                print(f'time steps horizon: {self.horizon}')
        if self.mode == 'tps':
            self.warper = tps.Warper(H=image_size, W=image_size, warpsd_all=0.00001,
                                     warpsd_subset=0.001, transsd=0.1, scalesd=0.1,
                                     rotsd=2, im1_multiplier=0.1, im1_multiplier_aff=0.1)
        else:
            self.warper = None
        data = np.load(path_to_npy)
        # train_size = int(0.9 * data.shape[0])
        # valid_size = data.shape[0] - train_size
        if train:
            self.data = data
            # self.data = data[:self.frames_per_video * 200]
            print(f'loaded data with shape: {self.data.shape}, size: {self.data.shape[0]}')
        else:
            self.data = data[:5000]
        self.image_size = image_size
        self.num_videos = self.data.shape[0] // self.frames_per_video
        self.video_as_index = video_as_index
        if transform is None:
            self.input_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            self.input_transform = transform

    def __getitem__(self, index):
        if not self.video_as_index:
            video_num = int(index / self.frames_per_video)
            video_start_idx = video_num * self.frames_per_video
            curr_idx = index % self.frames_per_video
            max_idx = min(video_start_idx + self.frames_per_video - 1, self.data.shape[0] - 1)
            global_idx = video_start_idx + curr_idx
            if self.mode == 'single':
                return self.input_transform(self.data[index])
            elif self.mode == 'frames':
                min_idx = min(video_start_idx, index - 1)
                if min_idx == video_start_idx:
                    im1 = self.input_transform(self.data[min_idx + 1])
                    im2 = self.input_transform(self.data[min_idx])
                else:
                    im1 = self.input_transform(self.data[min_idx])
                    im2 = self.input_transform(self.data[min_idx - 1])
                return im1, im2
            elif self.mode == 'horizon':
                images = []
                length = max_idx
                if (index + self.horizon) >= length:
                    slack = index + self.horizon - length
                    index = index - slack
                for i in range(self.horizon):
                    t = index + i
                    images.append(self.input_transform(self.data[t]))
                images = torch.stack(images, dim=0)
                return images
            elif self.mode == 'tps':
                im = self.input_transform(self.data[index])
                im = im * 255
                im2, im1, _, _, _, _ = self.warper(im)
                return im1 / 255, im2 / 255
            else:
                raise NotImplementedError
        else:
            video_num = index
            video_start_idx = video_num * self.frames_per_video
            max_idx = video_start_idx + self.frames_per_video - 1
            images = []
            length = max_idx
            frame_idx = video_start_idx
            actual_horizon = self.frames_per_video if ((frame_idx + self.horizon) >= length) else self.horizon
            for i in range(actual_horizon):
                t = frame_idx + i
                images.append(self.input_transform(self.data[t]))
            images = torch.stack(images, dim=0)
            return images

    def __len__(self):
        if not self.video_as_index:
            return self.data.shape[0]
        else:
            return self.num_videos


if __name__ == '__main__':
    path_to_img = '/media/newhd/data/clevrer/train/frames/'
    # prepare_numpy_file(path_to_img, image_size=128, frameskip=3, start_frame=26)
    test_epochs = False
    # load data
    path_to_npy = '/media/newhd/data/clevrer/valid/clevrer_img128np_fs3_valid.npy'
    mode = 'frames'
    horizon = 4
    train = True
    clevrer_ds = CLEVRERDataset(path_to_npy, mode=mode, train=train, horizon=horizon)
    clevrer_dl = DataLoader(clevrer_ds, shuffle=True, pin_memory=True, batch_size=5)
    batch = next(iter(clevrer_dl))
    if mode == 'single':
        im1 = batch[0]
    elif mode == 'frames' or mode == 'tps':
        im1 = batch[0][0]
        im2 = batch[1][0]

    if mode == 'single':
        print(im1.shape)
        img_np = im1.permute(1, 2, 0).data.cpu().numpy()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.imshow(img_np)
    elif mode == 'horizon':
        print(f'batch shape: {batch.shape}')
        images = batch[0]
        print(f'images shape: {images.shape}')
        fig = plt.figure(figsize=(8, 8))
        for i in range(images.shape[0]):
            ax = fig.add_subplot(1, horizon, i + 1)
            im = images[i]
            im_np = im.permute(1, 2, 0).data.cpu().numpy()
            ax.imshow(im_np)
            ax.set_title(f'im {i + 1}')
    else:
        print(f'im1: {im1.shape}, im2: {im2.shape}')
        im1_np = im1.permute(1, 2, 0).data.cpu().numpy()
        im2_np = im2.permute(1, 2, 0).data.cpu().numpy()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(im1_np)
        ax.set_title('im1')

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(im2_np)
        ax.set_title('im2 [t-1] or [tps]')
    plt.show()
    if test_epochs:
        from tqdm import tqdm

        pbar = tqdm(iterable=clevrer_dl)
        for batch in pbar:
            pass
        pbar.close()
