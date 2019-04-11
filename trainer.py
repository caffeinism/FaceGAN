import torch.nn as nn
from config import config
import torch.optim as optim
from torch.autograd import Variable
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.autograd as autograd
import torch.nn.init as init
import torch.nn.functional as F
from utils import Logger
import tf_recorder as tensorboard
from networks import define_D, define_G, TVLoss, FeatureL1Loss
import light_cnn
from copy import deepcopy
from dataset import Dataset

class Trainer:
    def __init__(self):
        self.generator = define_G(input_nc=3, output_nc=3, ngf=64, netG='unet_{}'.format(config.image_size), 
                                  norm='instance', nl='relu', use_dropout=False, init_type='xavier', gpu_ids=[0], upsample='residual')
        self.discriminator = define_D(input_nc=3 + 3, size=config.image_size, ndf=64, norm='instance',
                                      init_type='xavier', num_Ds=2, gpu_ids=[0])

        self.optimizer_g = optim.RMSprop(self.generator.parameters(), lr=config.lr, alpha=0.99, eps=1e-8)
        self.optimizer_d = optim.RMSprop(self.discriminator.parameters(), lr=config.lr, alpha=0.99, eps=1e-8)
        
        self.criterion_L1 = nn.L1Loss().cuda()
        self.criterion_TV = TVLoss().cuda()
        self.criterion_feature = FeatureL1Loss([0.5, 0.25, 0.375, 0.5, 1.0]).cuda()
        
        self.feature_extractor = light_cnn.LightCNN_9Layers()
        self.feature_extractor.train()
        self.feature_extractor.cuda()
        pretrained = torch.load(config.light_cnn)
        self.feature_extractor.load_state_dict(pretrained)
        toggle_grad(self.feature_extractor, False)

        self.reg_param = config.lambda_reg

        self.tb = tensorboard.tf_recorder()
        if config.resume: 
            self.load_model(config.resume)

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out, targets)

        return loss

    def generator_trainstep(self, input, input_landmark, target_landmark):
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        
        self.generator.train()
        self.discriminator.train()
                
        generator_hat = deepcopy(self.generator)
        toggle_grad(generator_hat, False)

        fake = self.generator(torch.cat([input, input_landmark], 1), target_landmark, detach_face=False, detach_pose=True)
        recon = generator_hat(torch.cat([fake, target_landmark], 1), input_landmark, detach_face=False, detach_pose=True)
        identity = self.generator(torch.cat([input, input_landmark], 1), input_landmark, detach_face=False, detach_pose=True)
        
        input_features, input_feature_vector = self.feature_extractor(input)

        _, fake_feature_vector = self.feature_extractor(fake)
        recon_features, recon_feature_vector = self.feature_extractor(recon)
        identity_features, _ = self.feature_extractor(identity)

        fake_pose = self.generator(torch.cat([input, input_landmark], 1), target_landmark, detach_face=True, detach_pose=False)
        d_fake = self.discriminator(torch.cat([fake_pose, target_landmark], 1))

        loss_g_dict = {
            'adv': self.compute_loss(d_fake, 1.0) * config.lambda_adv,
            'tv': self.criterion_TV(fake) * config.lambda_tv,
            'identity': self.criterion_feature(identity_features, input_features),
            'feature': self.criterion_L1(fake_feature_vector, input_feature_vector) * config.lambda_feature,
            'cycle1': self.criterion_feature(recon_features, input_features) * config.lambda_cycle,
            'cycle2': self.criterion_L1(recon_feature_vector, input_feature_vector) * config.lambda_feature,
        }

        loss_g = sum(loss_g_dict.values())

        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g_dict

    # https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
    def discriminator_trainstep(self, x_real, target, input_landmark, target_landmark):       
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)

        self.generator.train()
        self.discriminator.train()
        self.optimizer_d.zero_grad()

        # On real data
        target = torch.cat([target, target_landmark], 1)
        target.requires_grad_()

        d_real = self.discriminator(target)
        dloss_real = self.compute_loss(d_real, 1.0)
        dloss_real.backward(retain_graph=True)

        reg = self.reg_param * compute_grad2(d_real, target).mean()
        reg.backward()
        
        # On fake data
        with torch.no_grad():
            x_fake = self.generator(torch.cat([x_real, input_landmark], 1), target_landmark)

        x_fake.requires_grad_()
        d_fake = self.discriminator(torch.cat([x_fake, target_landmark], 1))
        dloss_fake = self.compute_loss(d_fake, 0.0)
        dloss_fake.backward()

        self.optimizer_d.step()

        # Output
        dloss = (dloss_real + dloss_fake)

        return {'loss_d': dloss, 'reg': reg} 

    def train(self, dataloader):
        print(config)
        
        logger = Logger(config.n_epoch, len(dataloader))
        global_iter = 0

        half = config.batch_size // 2
        for epoch in range(config.n_epoch):                                
            for i, (face, land) in enumerate(dataloader, 0):
                face = face.cuda()
                land = land.cuda()

                input = face[:half, ...]
                target = face[half:, ...]

                input_landmark = land[:half, ...]
                target_landmark = land[half:, ...]

                loss_d_dict = self.discriminator_trainstep(input, target, input_landmark, target_landmark)
                loss_g_dict = self.generator_trainstep(input, input_landmark, target_landmark)
                
                if global_iter % config.log_iter == 0:
                    self.generator.eval()
                    self.discriminator.eval()

                    with torch.no_grad():
                        fake = self.generator(torch.cat([target, target_landmark], 1), input_landmark)
                        
                        d_real = float(torch.sigmoid(self.discriminator(torch.cat([input, input_landmark], 1))).mean().item())
                        d_fake = float(torch.sigmoid(self.discriminator(torch.cat([fake, input_landmark], 1))).mean().item())

                    loss_dict = dict(**loss_g_dict, **loss_d_dict)
                    image_dict = {
                        'input': target,
                        'fake': fake,
                        'landmark': input_landmark,
                    }

                    logger.log(config.log_iter,
                               losses=loss_dict,
                               images=image_dict)
                    
                    if global_iter % (config.log_iter * 30) == 0:
                        print()
                        vutils.save_image(torch.cat([target[:8].data, fake[:8].data, input_landmark[:8].data], 0),
                                        '{}/result_epoch_{:03d}_iter_{:05d}.png'.format(config.result_dir, epoch, i),
                                        normalize=True, nrow=8)
                
                    self.tb.add_scalar('data/d_real', d_real)
                    self.tb.add_scalar('data/d_fake', d_fake)
                    self.tb.iter()
                global_iter += 1
            self.save_model(epoch)
    
    def save_model(self, epoch):
        save_dict = {
            'global_iter': self.tb.niter,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
        }
        torch.save(save_dict, '{}/epoch_{}.pth'.format(config.model_dir, epoch))

    def load_model(self, filename):
        model = torch.load(filename)
        self.generator.load_state_dict(model['generator'])
        self.discriminator.load_state_dict(model['discriminator'])
        self.optimizer_g.load_state_dict(model['optimizer_g'])
        self.optimizer_d.load_state_dict(model['optimizer_d'])
            

def main():
    dataset = Dataset(root=config.dataset_dir, land_root=config.landmark_dir, image_size=config.image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                             num_workers=config.n_cpu, pin_memory=True, drop_last=True)

    trainer = Trainer()
    trainer.train(dataloader)

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


if __name__ == '__main__':
    main()