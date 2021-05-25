import numpy as np
from PIL import Image as pimg

__all__ = ['StorePreds', 'StoreSubmissionPreds']


class StorePreds:
    def __init__(self, store_dir, to_img, to_color, step=0):
        self.store_dir = store_dir
        self.to_img = to_img
        self.to_color = to_color
        self.step = step
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __str__(self):
        return ''

    def __call__(self, pred, batch, additional):
        #b = self.to_img(batch)
        self.step += 1
        for p, name, subset in zip(pred, batch['name'], batch['subset']):
            store_img = np.concatenate([i.astype(np.uint8) for i in [self.to_color(p)]], axis=0)
            store_img = pimg.fromarray(store_img)
            store_img.thumbnail((960, 1344))
            if self.step in range(1,3122):
                store_img.save(f'{self.store_dir}ISA2/Highway/H1/{name}.jpg')
            if self.step in range(3123,4962):
                store_img.save(f'{self.store_dir}ISA2/Highway/H2/{name}.jpg')
            if self.step in range(4963,6787):
               store_img.save(f'{self.store_dir}ISA2/Urban/U1/{name}.jpg')
            if self.step in range(6788,8737):
                store_img.save(f'{self.store_dir}ISA2/Urban/U2/{name}.jpg')
            if self.step in range(8738,10419):
                store_img.save(f'{self.store_dir}ISA2/Urban/U3/{name}.jpg')
            #store_img.save(f'{self.store_dir}/ISA2/{name}.jpg')

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
