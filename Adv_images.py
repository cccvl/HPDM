import argparse
import torch
import kornia
import lpips
from utils import read_data, adv_save, load_weights
from utils import get_model
import model as models
from utils import data_preprocess
from tqdm import tqdm
from tools.cam_module import get_cam_mask
from pytorch_grad_cam import GradCAM
import numpy as np
from datetime import datetime
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))

def main(args):
    device = torch.device(args.device)

    resnet50_model = get_model('resnet50',device)
    resnet50_model = load_weights(resnet50_model, args.pretrained_resnet50_path, device)
    resnet50_model.eval()

    xception_model = get_model('xception',device)
    xception_model = load_weights(xception_model, args.pretrained_xception_path, device)
    xception_model.eval()

    efficient_model = get_model('efficientnet',device)
    efficient_model = load_weights(efficient_model, args.pretrained_efficient_path, device)
    efficient_model.eval()

    pretrained_G = models.Generator(3, 3).to(device)
    pretrained_G.load_state_dict(torch.load(args.pretrained_generator_path, map_location=device))
    pretrained_G.eval()

    test_images_path, test_images_label = read_data(args.test_path)
    test_data_set, test_loader = data_preprocess(test_images_path, test_images_label, args.image_size, args.batch_size)

    resnet50_cam = GradCAM(model=resnet50_model, target_layers=[resnet50_model.layer4[-1]])
    xception_cam = GradCAM(model=xception_model, target_layers=[xception_model.conv4])
    efficient_cam = GradCAM(model=efficient_model, target_layers=[efficient_model.features[-1]])

    for i, data in enumerate(tqdm(test_loader), 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)

        save_dir = '{}/advimages/'
        os.makedirs(save_dir, exist_ok=True)
        
        perturbation = pretrained_G(test_img)
        cam_soft_masks, cam_masks = get_cam_mask(test_img, resnet50_cam, xception_cam, efficient_cam)
        perturbation = torch.mul(perturbation, cam_masks.to(device))
        perturbation = torch.clamp(perturbation, -0.3, 0.3) 
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)

        adv_save(adv_img, save_dir, test_images_path, i*args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data-path', type=str,default="/home/wuzixiang/数据集/FFHQ_StyleGAN2/train/fake")

    parser.add_argument('--pretrained-resnet50-path', type=str,default='{}/trained_models/resnet50/best_epoch.pt'.format(parent_dir)) 
    parser.add_argument('--pretrained-xception-path', type=str,default='{}/trained_models/xception/best_epoch.pt'.format(parent_dir)) 
    parser.add_argument('--pretrained-efficient-path', type=str,default='{}/trained_models/efficientnet/best_epoch.pt'.format(parent_dir)) 
    
    parser.add_argument('--device', default='cuda:1')

    opt = parser.parse_args()

    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(1000)
    
    opt.pretrained_generator_path = '{}/OUTPUT/models/best_epoch.pth'.format(parent_dir)

    main(opt)
