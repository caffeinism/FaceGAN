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

def find_items(dir, class_to_idx, is_train):
    result = {idx: [] for idx in class_to_idx.values()}
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            if is_train and len(fnames) < 10:
                del result[class_to_idx[target]]
                del class_to_idx[target]
                print(target, 'is excluded')
                break
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(target, '.'.join(fname.split('.')[:-1]))
                    result[class_to_idx[target]].append(path)

    for idx, (key, value) in enumerate(sorted(result.items())):
        result[idx] = result[key]
        if idx != key:
            del result[key]

    return result

def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB' if rgb else 'L')


class Dataset(data.Dataset):
    def __init__(self, root, land_root, image_size, loader=pil_loader, is_train=True):
        _, class_to_idx = find_classes(land_root)
        class_item_dict = find_items(land_root, class_to_idx, is_train)

        if len(class_item_dict) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.land_root = land_root
        self.loader = loader


        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.class_item_dict = class_item_dict

    def __getitem__(self, index):
        path_samples = random.sample(self.class_item_dict[index], 2)

        image_paths = [os.path.join(self.root, it + '.jpg') for it in path_samples]
        landmark_paths = [os.path.join(self.land_root, it + '.png') for it in path_samples]

        images = [self.loader(it) for it in image_paths]
        landmarks = [self.loader(it) for it in landmark_paths]


        images = [self.transform(it) for it in images]
        landmarks = [self.transform(it) for it in landmarks]
        
        return [
            (img, landmark) for img, landmark in zip(images, landmarks)
        ]

    def __len__(self):
        return len(self.class_item_dict)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str