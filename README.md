# deep-latent-particles-pytorch
Official PyTorch implementation of the paper "Unsupervised Image Representation Learning with Deep Latent Particles"

<h1 align="center">
  <br>
	Paper Title
  <br>
</h1>
  <p align="center">
    <a href="https://taldatech.github.io">Tal Daniel</a> •
    <a href="https://avivt.github.io/avivt/">Aviv Tamar</a>

  </p>
<h4 align="center">Official repository of the paper</h4>

<h4 align="center">Venue</h4>

<h4 align="center"><a href="https://taldatech.github.io/">Project Website</a> • <a href="">Video</a></h4>

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/soft-intro-vae-pytorch"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</h4>


<p align="center">
  <img src="https://github.com/taldatech/soft-intro-vae-web/raw/main/assets/ffhq_samples.png" height="120">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/celebahq_recons.png" height="120">
</p>
<p align="center">
  <img src="https://github.com/taldatech/soft-intro-vae-web/raw/main/assets/3d_plane_to_car.gif" height="100">
  <img src="https://raw.githubusercontent.com/taldatech/soft-intro-vae-web/main/assets/density_plot_png_f.PNG" height="100">
</p>

# Deep Latent Particles

> **Unsupervised Image Representation Learning with Deep Latent Particles**<br>
> Tal Daniel, Aviv Tamar<br>
>
> **Abstract:** *The recently introduced introspective variational autoencoder (IntroVAE) exhibits outstanding image generations, and allows for amortized inference using an image encoder. The main idea in IntroVAE is to train a VAE adversarially, using the VAE encoder to discriminate between generated and real data samples. However, the original IntroVAE loss function relied on a particular hinge-loss formulation that is very hard to stabilize in practice, and its theoretical convergence analysis ignored important terms in the loss. In this work, we take a step towards better understanding of the IntroVAE model, its practical implementation, and its applications. We propose the Soft-IntroVAE, a modified IntroVAE that replaces the hinge-loss terms with a smooth exponential loss on generated samples. This change significantly improves training stability, and also enables theoretical analysis of the complete algorithm. Interestingly, we show that the IntroVAE converges to a distribution that minimizes a sum of KL distance from the data distribution and an entropy term. We discuss the implications of this result, and demonstrate that it induces competitive image generation and reconstruction. Finally, we describe two applications of Soft-IntroVAE to unsupervised image translation and out-of-distribution detection, and demonstrate compelling results.*

## Citation
Daniel, Tal, and Aviv Tamar. "Soft-IntroVAE: Analyzing and Improving the Introspective Variational Autoencoder." arXiv preprint arXiv:2012.13253 (2020).
>
    @InProceedings{Daniel_2021_CVPR,
    author    = {Daniel, Tal and Tamar, Aviv},
    title     = {Soft-IntroVAE: Analyzing and Improving the Introspective Variational Autoencoder},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {4391-4400}
}

<h4 align="center">Preprint on ArXiv: <a href="https://arxiv.org/abs/2012.13253">Soon</a></h4>


- [soft-intro-vae-pytorch](#soft-intro-vae-pytorch)
- [Soft-IntroVAE](#soft-introvae)
  * [Citation](#citation)
  * [Prerequisites](#prerequisites)
  * [Repository Organization](#repository-organization)
  * [Credits](#credits)
    

## Prerequisites

* For your convenience, we provide an `environemnt.yml` file which installs the required packages in a `conda` environment name `torch`.
    * Use the terminal or an Anaconda Prompt and run the following command `conda env create -f environment.yml`.


| Library           | Version          |
|-------------------|------------------|
| `Python`          | `3.7 (Anaconda)` |
| `torch`           | >= `1.7.1`       |
| `torch_geometric` | >= `1.7.1`       |
| `torchvision`     | >= `0.4`         |
| `matplotlib`      | >= `2.2.2`       |
| `numpy`           | >= `1.17`        |
| `py-opencv`       | >= `3.4.2`       |
| `tqdm`            | >= `4.36.1`      |
| `scipy`           | >= `1.3.1`       |
| `scikit-image`    | >= `0.18.1`      |
| `accelerate`      | >= `0.3.0`       |

## Pretrained Models

* We provide pre-trained checkpoints for the 3 datasets we used in the paper.
* All model checkpoints should be placed inside the `/checkpoints` directory.
* The interactive demo will use these checkpoints.

| Dataset           | Filename                           | Link                                                                                |
|-------------------|------------------------------------|--------------------------------------------------------------------------------------|
| CelebA (128x128)  | `dlp_celeba_gauss_pointnetpp_feat.pth` | [MEGA.co.nz](https://mega.nz/file/ZAkiDSIQ#ndtlzAPwG42TEGZmuADAzR1Wo0AZx2k__qyWUfcOkQc)|
| Traffic (128x128) | `dlp_traffic_gauss_pointnetpp.pth` | [MEGA.co.nz](https://mega.nz/file/VINjRZCL#rJ25UPXlYJUxWPaP7gDEbxjVZaayey5JB6x9P5Z__CU)|
| CLEVRER (128x128) | `dlp_clevrer_gauss_pointnetpp.pth` | [MEGA.co.nz](https://mega.nz/file/9cN0HAYQ#K9AvKsWemA5hvk9WleleautIdQu2Euezf8UOI7aKUtE)|

## Interactive Demo

* We designed a simple `matplotlib` interactive GUI to plot and control the particles. 
* The demo is a standalone and does not require to download the original datasets.
* We provide sample images inside `/checkpoints/sample_images/` which will be used.


To run the demo (after downloading the checkpoints): `python interactive_demo_dlp.py --help`
* `-d`: dataset to use: [`celeba`, `traffic`, `clevrer`]
* `-i`: index of the image to use inside `/checkpoints/sample_images/`

Examples:
* `python interactive_demo_dlp.py -d celeba -i 2`
* `python interactive_demo_dlp.py -d traffic -i 0`
* `python interactive_demo_dlp.py -d clevrer -i 0`

You can modify `interactive_demo_dlp.py` to add additional datasets.

## Datasets

* **CelebA**: we follow [DVE](https://github.com/jamt9000/DVE):
  * [Download](https://github.com/jamt9000/DVE/blob/master/misc/datasets/celeba/README.md) the dataset from this [link](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/datasets/celeba.tar.gz).
  * The pre-processing is described in `datasets/celeba_dataset.py`.
* **CLEVRER**: download the training and validation videos from [here](http://clevrer.csail.mit.edu/):
  * [Training Videos](http://data.csail.mit.edu/clevrer/videos/train/video_train.zip), [Validation Videos](http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip)
  * Follow the pre-processing in `datasets/clevrer_ds.py` (`prepare_numpy_file(path_to_img, image_size=128, frameskip=3, start_frame=26`)
* **Traffic**: this is a self-collected dataset, please contact us if you wish to use it.
* **Shapes**: this dataset is generated automatically in each run for simplicity, see `generate_shape_dataset_torch()` in `datasets/shapes_ds.py`.

## Training 

* Single-GPU machines: `python train_dlp.py --help`
* Multi-GPU machines: `accelerate --config_file ./accel_conf.json train_dlp_accelerate.py --help`


You should run the `train_dlp.py` or `train_dlp_accelerate.py` files with the following arguments:

|Argument                 | Description                                 |Legal Values |
|-------------------------|---------------------------------------------|-------------|
|-h, --help       | shows arguments description             			| 			|
|-d, --dataset     | dataset to train on 				               	|str: 'cifar10', 'mnist', 'fmnist', 'svhn', 'monsters128', 'celeb128', 'celeb256', 'celeb1024'	|
|-n, --num_epochs	| total number of epochs to run			| int: default=250|
|-z, --z_dim| latent dimensions										| int: default=128|
|-s, --seed| random state to use. for random: -1 						| int: -1 , 0, 1, 2 ,....|
|-v, --num_vae| number of iterations for vanilla vae training 				| int: default=0|
|-l, --lr| learning rate 												| float: defalut=2e-4 |
|-r, --beta_rec | beta coefficient for the reconstruction loss |float: default=1.0|
|-k, --beta_kl| beta coefficient for the kl divergence							| float: default=1.0|
|-e, --beta_neg| beta coefficient for the kl divergence in the expELBO function | float: default=256.0|
|-g, --gamma_r| coefficient for the reconstruction loss for fake data in the decoder		| float: default=1e-8|
|-b, --batch_size| batch size 											| int: default=32 |
|-p, --pretrained     | path to pretrained model, to continue training	 	|str: default="None"	|
|-c, --device| device: -1 for cpu, 0 and up for specific cuda device						|int: default=-1|
|-f, --fid| if specified, FID wil be calculated during training				|bool: default=False|

Examples:

* Single-GPU:

`python main.py --dataset cifar10 --device 0 --lr 2e-4 --num_epochs 250 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 256 --z_dim 128 --batch_size 32`

`python main.py --dataset mnist --device 0 --lr 2e-4 --num_epochs 200 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 256 --z_dim 32 --batch_size 128`

* Multi-GPU:

## Evaluation of Unsupervised Keypoint Regression on CelebA



## Recommended Hyper-parameters

|Dataset | `beta_kl` | `beta_rec`| `beta_neg`|`z_dim`|`batch_size`|
|------------|------|----|---|----|---|
|CIFAR10 (`cifar10`)|1.0|1.0| 256|128| 32|
|SVHN (`svhn`)|1.0|1.0| 256|128| 32|
|MNIST (`mnist`)|1.0|1.0|256|32|128|
|FashionMNIST (`fmnist`)|1.0|1.0|256|32|128|
|Monsters (`monsters128`)|0.2|0.2|256|128|16|
|CelebA (`celeb256`)|0.5|1.0|1024|256|8|

## Repository Organization

|File name         | Content |
|----------------------|------|
|`/soft_intro_vae`| directory containing implementation for image data|
|`/soft_intro_vae_2d`| directory containing implementations for 2D datasets|
|`/soft_intro_vae_3d`| directory containing implementations for 3D point clouds data|
|`/soft_intro_vae_bootstrap`| directory containing implementation for image data using bootstrapping (using a target decoder)|
|`/style_soft_intro_vae`| directory containing implementation for image data using ALAE's style-based architecture|
|`/soft_intro_vae_tutorials`| directory containing Jupyter Noteboook tutorials for the various types of Soft-IntroVAE|



## Credits
* Related papers/repositories
