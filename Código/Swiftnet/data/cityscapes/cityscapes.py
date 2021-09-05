from torch.utils.data import Dataset
from pathlib import Path

from .labels import labels
import os

home = Path.home()

class_info = [label.name for label in labels if label.ignoreInEval is False]
color_info = [label.color for label in labels if label.ignoreInEval is False]

color_info += [[0, 0, 0]]

map_to_id = {}
inst_map_to_id = {}
i, j = 0, 0
for label in labels:
    if label.ignoreInEval is False:
        map_to_id[label.id] = i
        i += 1
        if label.hasInstances is True:
            inst_map_to_id[label.id] = j
            j += 1

id_to_map = {id: i for i, id in map_to_id.items()}
inst_id_to_map = {id: i for i, id in inst_map_to_id.items()}


class Cityscapes(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 19

    map_to_id = map_to_id
    id_to_map = id_to_map

    inst_map_to_id = inst_map_to_id
    inst_id_to_map = inst_id_to_map

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root: home, transforms: lambda x: x, subset='train', open_depth=False, labels_dir='labels', epoch=None):
        self.root = home
        self.subset = subset
        self.images_dir = self.root / 'Desktop' / 'Trabajo-Final' / 'Código' / 'Swiftnet' / 'datasets' / 'ISA2'
        self.images = list(sorted(self.images_dir.glob('Urban/U3/*.jpeg')))
        #self.images = list(sorted(self.images_dir.glob('*/*/*.jpeg')))
        self.transforms = transforms
        self.epoch = epoch
        print(f'Num images: {len(self)}')


    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'image': self.images[item],
            'name': self.images[item].stem,
            'subset': self.subset,
        }
        #if self.has_labels:
        #    ret_dict['labels'] = self.labels[item]
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)
        return self.transforms(ret_dict)

'''
        self.images = []
        files = os.listdir(self.images_dir)
        dirs = []
        subdirs_H = []
        subdirs_U = []
        i = 0
        for f in files:
            if os.path.isdir(os.path.join(self.images_dir, f)):
                dirs.append(f)
        for d in dirs:
            subfiles = os.listdir(os.path.join(self.images_dir, d))
            subfiles.sort()
            i += 1
            for s in subfiles:
                if os.path.isdir(os.path.join(self.images_dir, d, s)) and i is 1:
                    subdirs_H.append(s)
                if os.path.isdir(os.path.join(self.images_dir, d, s)) and i is len(dirs):
                    subdirs_U.append(s)
        #self.labels_dir = self.root / 'Desktop' / 'Trabajo-Final' / 'Código' / 'Swiftnet' / 'datasets' / 'Cityscapes' / labels_dir / subset
        #self.depth_dir = self.root / 'Desktop' / 'Trabajo-Final' / 'Código' / 'Swiftnet' / 'datasets' / 'Cityscapes' / 'depth' / subset
        self.subset = subset
        self.has_labels = subset != 'test'
        self.open_depth = open_depth
        images_subdirs = []
        x = 0
        for d in dirs:
            x += 1
            if x is 1:
                for s in subdirs_H:
                    images_subdirs = list(sorted(self.images_dir.glob(f'{d}/{s}/*.jpeg')))
                    for p in range(0, len(images_subdirs)):
                        self.images.append(images_subdirs[p])
            if x is len(dirs):
                for s in subdirs_U:
                    images_subdirs = list(sorted(self.images_dir.glob(f'{d}/{s}/*.jpeg')))
                    for p in range(0, len(images_subdirs)):
                        self.images.append(images_subdirs[p])
'''
