# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py

import torch.utils.data as data
import sys
import os
import math
import random
import torch
from PIL import Image
import torchvision.transforms as transforms

def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def find_items(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(target, os.path.splitext(fname)[0])
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB' if rgb else 'L')


# Dataset for (face, landmark) pair
class Dataset(data.Dataset):
    def __init__(self, root, land_root, image_size, loader=pil_loader):
        extensions=IMG_EXTENSIONS
        
        classes, class_to_idx = find_classes(land_root)
        samples = find_items(land_root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + land_root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.land_root = land_root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        path, _ = self.samples[index]

        face_path = os.path.join(self.root, path) + '.jpg'
        land_path = os.path.join(self.land_root, path) + '.png'
        
        face_sample = self.loader(face_path)
        land_sample = self.loader(land_path)
        
        face_sample = self.transform(face_sample)
        land_sample = self.transform(land_sample)

        return face_sample, land_sample

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str