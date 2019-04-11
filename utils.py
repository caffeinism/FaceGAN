import time
import datetime
import sys

import torch
from visdom import Visdom
import numpy as np
from torch.nn import init
import torchvision.transforms as transforms
import torch.nn.functional as F

to_pil = transforms.ToPILImage()

def tensor2image(tensor):
    image = 255.0 * tensor[:4, ...].cpu().float().numpy()
    
    return image.astype(np.uint8)
    
# https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/utils.py
class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, log_iter, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.images(tensor2image(tensor.data), nrow=4,
                                                                 opts={'title': image_name})
            else:
                self.viz.images(tensor2image(tensor.data), nrow=4, win=self.image_windows[image_name],
                                opts={'title': image_name})

        # End of epoch
        if (self.batch >= self.batches_epoch):
            # Plot losses
            for loss_name, loss in self.losses.items():
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = self.batch % self.batches_epoch + log_iter
            sys.stdout.write('\n')
        else:
            self.batch += log_iter