import argparse

parser = argparse.ArgumentParser("DCGAN")
parser.add_argument('--dataset_dir', type=str, default='../VGGFace2/train_normal/')             # face dataset directory
parser.add_argument('--landmark_dir', type=str, default='../VGGFace2/train_normal_landmark/')   # landmark dataset directory
parser.add_argument('--result_dir', type=str, default='result')                                 # log image save directory
parser.add_argument('--model_dir', type=str, default='model')                                   # trained model save directory
parser.add_argument('--light_cnn', type=str, default='models/LightCNN_9Layers.pth')             # pretrained light cnn model path
parser.add_argument('--resume', type=str, default='')                                           # load model path if exist

parser.add_argument('--image_size', type=int, default=128)  # image size
parser.add_argument('--batch_size', type=int, default=16)   # batch size
parser.add_argument('--n_epoch', type=int, default=100)     # epoch size
parser.add_argument('--n_cpu', type=int, default=0)         # num of process(for use worker)
parser.add_argument('--log_iter', type=int, default=10)     # print log message and save image per log_iter
parser.add_argument('--nz', type=int, default=256)          # noise dimension
parser.add_argument('--nc', type=int, default=3)            # input and out channel
parser.add_argument('--ndf', type=int, default=64)          # number of discriminator's feature map dimension
parser.add_argument('--ngf', type=int, default=64)          # number of generator's feature map dimension

parser.add_argument('--lr', type=float, default=0.0001)             # learning rate
parser.add_argument('--lambda_adv', type=float, default=1.0)        # weight for adversarial loss
parser.add_argument('--lambda_identity', type=float, default=0.003) # weight for identity loss
parser.add_argument('--lambda_cycle', type=float, default=1.)       # weight for feature cycle loss
parser.add_argument('--lambda_tv', type=float, default=0.00001)     # weight for tv loss
parser.add_argument('--lambda_feature', type=float, default=0.0005) # weight for feature loss
parser.add_argument('--lambda_reg', type=float, default=100.0)      # weight for gradient regularization 

config, _ = parser.parse_known_args()