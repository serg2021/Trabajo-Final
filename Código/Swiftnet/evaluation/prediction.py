import numpy as np
import os
from pathlib import Path
import glob

import scipy.io
from scipy.io import savemat, loadmat
from PIL import Image as pimg

__all__ = ['StorePreds', 'StoreSubmissionPreds']
home = Path.home()

class StorePreds:
    def __init__(self, store_dir, to_img, to_color, step=0):
        self.store_dir = store_dir
        self.to_img = to_img
        self.to_color = to_color
        self.step = step
        self.t = 0
        self.d = 0
        self.s = 0
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __str__(self):
        return ''

    def __call__(self, pred, batch, additional):
        #b = self.to_img(batch)
        self.root = home
        self.images_dir = self.root / 'Desktop' / 'Trabajo-Final' / 'Código' / 'Swiftnet' / 'datasets' / 'ISA2'
        '''files = os.listdir(self.images_dir)
        dirs = []
        subdirs = []
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
                    subdirs.append(s)
                if os.path.isdir(os.path.join(self.images_dir, d, s)) and i is len(dirs):
                    subdirs.append(s)
        self.step += 1'''
        for p, name, subset in zip(pred, batch['name'], batch['subset']):
            store_img = np.concatenate([i.astype(np.uint8) for i in [self.to_color(p)]], axis=0)
            store_img = pimg.fromarray(store_img)
            store_img.thumbnail((960, 1344))
            scipy.io.savemat(f'{self.store_dir}/ISA2/{name}.mat', {"name": name, "data" : p})
            '''images_subdir = len(glob.glob(f'{self.images_dir}/{dirs[self.d]}/{subdirs[self.s]}/*.jpeg'))
            self.t += 1
            if self.d is 0:
                if self.t <= images_subdir:
                    scipy.io.savemat(f'{self.store_dir}/ISA2/{dirs[self.d]}/{subdirs[self.s]}/{name}.mat', {"name": name, "data" : p})
                    #store_img.save(f'{self.store_dir}/ISA2/{dirs[self.d]}/{subdirs[self.s]}/{name}.png')
                if self.t > images_subdir:
                    self.t = 1
                    self.s += 1
                    if self.s is 2:
                        self.d = 1
                    scipy.io.savemat(f'{self.store_dir}/ISA2/{dirs[self.d]}/{subdirs[self.s]}/{name}.mat',
                                     {"name": name, "data": p})
                    #store_img.save(f'{self.store_dir}/ISA2/{dirs[self.d]}/{subdirs[self.s]}/{name}.png')
            if self.d is 1:
                if self.t <= images_subdir:
                    scipy.io.savemat(f'{self.store_dir}/ISA2/{dirs[self.d]}/{subdirs[self.s]}/{name}.mat',
                                     {"name": name, "data": p})
                    #store_img.save(f'{self.store_dir}/ISA2/{dirs[self.d]}/{subdirs[self.s]}/{name}.png')
                if self.t > images_subdir:
                    self.t = 1
                    self.s += 1
                    scipy.io.savemat(f'{self.store_dir}/ISA2/{dirs[self.d]}/{subdirs[self.s]}/{name}.mat',
                                     {"name": name, "data": p})
                    #store_img.save(f'{self.store_dir}/ISA2/{dirs[self.d]}/{subdirs[self.s]}/{name}.png')'''
            '''if self.step in range(1,3122):
                store_img.save(f'{self.store_dir}ISA2/Highway/H1/{name}.jpg')
            if self.step in range(3123,4962):
                store_img.save(f'{self.store_dir}ISA2/Highway/H2/{name}.jpg')
            if self.step in range(4963,6787):
               store_img.save(f'{self.store_dir}ISA2/Urban/U1/{name}.jpg')
            if self.step in range(6788,8737):
                store_img.save(f'{self.store_dir}ISA2/Urban/U2/{name}.jpg')
            if self.step in range(8738,10419):
                store_img.save(f'{self.store_dir}ISA2/Urban/U3/{name}.jpg')'''
            #store_img.save(f'{self.store_dir}//{name}.jpg')

class StoreSubmissionPreds:
    def __init__(self, store_dir, remap, to_color=None, store_dir_color=None):
        self.store_dir = store_dir
        self.store_dir_color = store_dir_color
        self.to_color = to_color
        self.remap = remap

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __str__(self):
        return ''

    def __call__(self, pred, batch, additional):
        for p, name in zip(pred.astype(np.uint8), batch['name']):
            pimg.fromarray(self.remap(p)).save(f'{self.store_dir}/{name}.png')
            pimg.fromarray(self.to_color(p)).save(f'{self.store_dir_color}/{name}.png')
