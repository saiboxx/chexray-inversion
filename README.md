# Implicit Embeddings via GAN Inversion for High Resolution Chest Radiographs

Code for the paper [Implicit Embeddings via GAN Inversion for High Resolution Chest Radiographs](https://link.springer.com/chapter/10.1007/978-3-031-25046-0_3)
at the *First MICCAI Workshop on Medical Applications with Disentanglements 2022*.

>Generative models allow for the creation of highly realistic artificial samples, 
> opening up promising applications in medical imaging. In this work, we propose a 
> multi-stage encoder-based approach to invert the generator of a 
> generative adversarial network (GAN) for high resolution chest radiographs. 
> This gives direct access to its implicitly formed latent space, makes generative 
> models more accessible to researchers, and enables to apply generative techniques 
> to actual patient’s images. We investigate various applications for this embedding, 
> including image compression, disentanglement in the encoded dataset, 
> guided image manipulation, and creation of stylized samples. 
> We find that this type of GAN inversion is a promising research direction in 
> the domain of chest radiograph modeling and opens up new ways to combine realistic 
> X-ray sample synthesis with radiological image analysis.

<p align="center">
<img src=assets/inversion_grid.png />
</p>

## Requirements:

An important building block of our pipeline is the progressive growing generator of
[Segal et al.](https://link.springer.com/article/10.1007/s42979-021-00720-7).
However, it seems that the weights of the generator were removed from 
their [repository](https://github.com/BradSegal/CXR_PGGAN) and 
we are not in the position to release it again.

The data used for training is the NIH Chest X-ray 14 dataset, which is available
[here](https://nihcc.app.box.com/v/ChestXray-NIHCC). Our default path for the dataset
is assumed to be `data/chex-ray14`.

Required packages are listed in the `requirements.txt` file.

The E4E loss uses a pretrained resnet-50 model using mocov2 in the similarity.
It can be downloaded from [here](https://drive.google.com/file/d/18rLcNGdteX5LwT7sv_F7HWr12HpVEzVe/view?usp=sharing).
The default path is assumed to be `models/mocov2.pt`


## Training

The training routines are optimized to work with DDP in a SLURM and will consume 
all available GPUs in the allocated instance.

## Stage 1: Bootstrapped Training

The run is configured via the config file `configs/inv_boot.yml`.
Training is started via:

```shell
python scripts/01_train_boot.py
```


## Stage 2: Dataset Training

The run is configured via the config file `configs/inv_fine.yml`.
Training is started via:

```shell
python scripts/02_train_fine.py
```

Have a look at `scripts/03_convert_basic.py` to convert the dataset into latent space
using a single encoder pass.

## Stage 3: Iterative Optimization

Iterative optimization can be very time and resource consuming.
In a lot of cases, the single encoder pass is good enough.
The per-sample optimization procedure can be triggered by:

```shell
python scripts/04_convert_optim.py
```

## Notebook

The repository contains a small showcasing example in `notebooks`, which illustrates
how to use the model objects.

## Models

**TBD SOON**

## Citation

If you use this code for your research, please cite our paper [Implicit Embeddings via GAN Inversion for High Resolution Chest Radiographs](https://link.springer.com/chapter/10.1007/978-3-031-25046-0_3):

```
@inproceedings{weber2023implicit,
  title={Implicit Embeddings via GAN Inversion for High Resolution Chest Radiographs},
  author={Weber, Tobias and Ingrisch, Michael and Bischl, Bernd and R{\"u}gamer, David},
  journal={Medical Applications with Disentanglements: First MICCAI Workshop, MAD 2022, Held in Conjunction with MICCAI 2022},
  pages={22--32},
  year={2023},
  organization={Springer}
}
```
