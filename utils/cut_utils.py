import torch

import argparse
import os
import sys
import time
#CUT_path = 
#sys.path.insert(1, CUT_path)

import numpy as np
from PIL import Image
from tqdm import tqdm


from data.base_dataset import get_transform
from data import create_dataset
from models_cut import create_model
import util.util as util
from util.visualizer import Visualizer
from torchvision import transforms

def get_CUT_args(exp_name: str, gpu_ids: list, cut_path: str='./modules/CUT/'):
	parser = argparse.ArgumentParser(description='CUT')

	# base_options
	parser.add_argument('--model', type=str, default='cut', help='chooses which model to use.')
	parser.add_argument('--gpu_ids', type=list, default=gpu_ids, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
	parser.add_argument('--isTrain', type=bool, default=False)
	parser.add_argument('--checkpoints_dir', type=str, default=os.path.join(cut_path, 'checkpoints'), help='models are saved here')

	# model parameters
	parser.add_argument('--name', type=str, default=exp_name, help='name of the experiment. It decides where to store samples and models')
	parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
	parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
	parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
	parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
	parser.add_argument('--netD', type=str, default='basic', choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'], help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
	parser.add_argument('--netG', type=str, default='resnet_9blocks', choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], help='specify generator architecture')
	parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
	parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
	parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for D')
	parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
	parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
	parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True,
	                    help='no dropout for the generator')
	parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
	parser.add_argument('--no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
	# dataset parameters
	parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
	parser.add_argument('--no_flip', action='store_true', default=True, help='if specified, do not flip the images for data augmentation')
	parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')

	# additional parameters
	parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
	parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
	parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

	# cut_model
	parser.add_argument('--CUT_mode', type=str, default="FastCUT", choices='(CUT, cut, FastCUT, fastcut)')
	parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
	parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
	parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
	parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
	parser.add_argument('--nce_includes_all_negatives_from_minibatch',
	                    type=util.str2bool, nargs='?', const=True, default=False,
	                    help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
	parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
	parser.add_argument('--netF_nc', type=int, default=256)
	parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
	parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
	parser.add_argument('--flip_equivariance',
	                    type=util.str2bool, nargs='?', const=True, default=False,
	                    help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
	opt, _ = parser.parse_known_args()

	# Set default parameters for CUT and FastCUT
	if opt.CUT_mode.lower() == "cut":
	    parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
	elif opt.CUT_mode.lower() == "fastcut":
	    parser.set_defaults(
	        nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
	        n_epochs=150, n_epochs_decay=50
	    )
	else:
	    raise ValueError(opt.CUT_mode)

	return opt

def get_CUT_norm(exp_name: str, gpu_ids: list, cut_path: str='./modules/CUT/'):
	opt = get_CUT_args(exp_name=exp_name, gpu_ids=gpu_ids, cut_path=cut_path)
	model = create_model(opt)
	model.setup(opt)
	cut_transform = get_transform(opt)
	return model, cut_transform

def tensor2rgb(input_image):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        
        image_tensor = image_tensor.clamp(-1.0, 1.0).cpu().float()  # convert it into a numpy array
        image_tensor = (image_tensor + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_tensor

class Batch_ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    def __init__(self):
        self.max = 255
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        return tensor.float().div_(self.max)
    
class Batch_Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor

def batch_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([
        Batch_ToTensor(),
        Batch_Normalize(mean = mean, std = std),
    ])
    return trnsfrms_val