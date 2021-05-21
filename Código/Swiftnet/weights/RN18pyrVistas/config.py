import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from torchvision.transforms import Compose
import torch.optim as optim
from pathlib import Path
import os
import numpy as np

from models.semseg import SemsegModel
from models.cascade_down_no_spp.resnet_conv_pyramid import *
from models.loss import BoundaryAwareFocalLoss
from data.transform import *
from data.vistas.vistas import Vistas
from evaluation import StorePreds
from multiprocessing import Value
import pickle

from models.util import get_n_params

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
root = Path('/home/morsic/datasets/vistas')

evaluating = True 

scale = 1
mean = [123.68, 116.779, 103.939]
std = [70.59564226, 68.52497082, 71.41913876]
mean_rgb = tuple(np.uint8(scale * np.array(mean)))
random_crop_size = 768

class_info = Vistas.class_info
color_info = Vistas.color_info
num_classes = Vistas.num_classes
ignore_id = Vistas.num_classes
mapping = {**{i: i for i in range(Vistas.num_classes)}, **{i: ignore_id for i in range(63, 66)}}

output_stride = 4

num_levels = 3

eval_each = 1
dist_trans_bins = (12, 38, 89)
# dist_trans_bins = (4, 16, 64)
dist_trans_alphas = (8., 4., 2., 1.)

class_incidence_train = np.load(root / 'class_incidence_train.npy')
with (root / 'class_instances_train.pkl').open('rb') as f:
    class_instances_train = pickle.load(f)

epochs = 100
epoch = Value('d', 0)

sampling_strategy_train, weights = create_class_uniform_strategy(class_instances_train, class_incidence_train, epochs)

trans_val = Compose(
    [Open(),
     RemapLabels(mapping, ignore_id=ignore_id, total=ignore_id),
     ResizeLongerSide(1920),
     Pad((1920, 1920), ignore_id=ignore_id, mean=mean_rgb),
     Tensor(),
     SetTargetSize(None, None),
     ]
)

if evaluating:
    trans_train = trans_val
else:
    trans_train = Compose(
        [Open(copy_labels=False),
         ResizeLongerSide(1920),
         RemapLabels(mapping, ignore_id=ignore_id, total=ignore_id),
         RandomFlip(),
         ClassUniformSquareCropAndScale(random_crop_size, mean_rgb, ignore_id, sampling_strategy_train, class_instances_train),
         LabelDistanceTransform(num_classes=num_classes, ignore_id=ignore_id, reduce=True, bins=dist_trans_bins, alphas=dist_trans_alphas),
         Tensor(),
         SetTargetSize(None, None),
         ])

dataset_train = Vistas(root, transforms=trans_train, subset='training', epoch=epoch)
dataset_val = Vistas(root, transforms=trans_val, subset='validation', epoch=epoch)

backbone = resnet18(pretrained=True,
                    pyramid_levels=num_levels,
                    k_upsample=3,
                    scale=scale,
                    mean=mean,
                    std=std,
                    k_bneck=1,
                    output_stride=output_stride,
                    efficient=False)
model = SemsegModel(backbone, num_classes, k=1, bias=True)
if evaluating:
    model.load_state_dict(torch.load(f'{dir_path}/stored/model_best.pt'), strict=False)
else:
    model.criterion = BoundaryAwareFocalLoss(gamma=.5, num_classes=num_classes, ignore_id=ignore_id)

bn_count = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        bn_count += 1
print(f'Num BN layers: {bn_count}')

print(f'Upsample modules:\n{model.backbone.upsample_blends}')

if not evaluating:
    lr = 8e-4
    lr_min = 1e-6
    fine_tune_factor = 4
    weight_decay = 1e-4

    optim_params = [
        {'params': model.random_init_params(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.fine_tune_params(), 'lr': lr / fine_tune_factor,
         'weight_decay': weight_decay / fine_tune_factor},
    ]

    optimizer = optim.Adam(optim_params, betas=(0.9, 0.99))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr_min)

batch_size = bs = 8
print(f'Batch size: {bs}')
nw = 4

subset_sampler_train = WeightedRandomSampler(weights=weights, num_samples=len(dataset_train))
# subset_sampler_train = None

loader_val = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate, num_workers=nw)
if evaluating:
    loader_train = DataLoader(dataset_train, batch_size=1, collate_fn=custom_collate, num_workers=nw)
else:
    loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=nw, pin_memory=True,
                              drop_last=True, collate_fn=custom_collate, sampler=subset_sampler_train)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')

eval_observers = [] 
if False and evaluating:
    eval_loaders = [(loader_val, 'validation'), (loader_train, 'training')]
    store_dir = f'{dir_path}/out/'
    for d in ['', 'validation', 'training']:
        os.makedirs(store_dir + d, exist_ok=True)
    to_color = ColorizeLabels(color_info)
    to_image = Compose([Numpy(), to_color])
    eval_observers += [StorePreds(store_dir, to_image, to_color)]
