import torch
from advGAN import AdvGAN_Attack
from utils import get_model,load_weights,read_data,data_preprocess
import argparse
from pytorch_grad_cam import GradCAM
import numpy as np
from datetime import datetime
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str,default="/home/wuzixiang/数据集/FFHQ_StyleGAN2/train/fake")

parser.add_argument('--pretrained-resnet50-path', type=str,default='{}/trained_models/resnet50/best_epoch.pt'.format(parent_dir)) 
parser.add_argument('--pretrained-xception-path', type=str,default='{}/trained_models/xception/best_epoch.pt'.format(parent_dir)) 
parser.add_argument('--pretrained-efficient-path', type=str,default='{}/trained_models/efficientnet/best_epoch.pt'.format(parent_dir)) 
args = parser.parse_args()

torch.manual_seed(1000)
torch.cuda.manual_seed_all(1000)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1000)

image_nc=3
epochs = 20
batch_size = 30
BOX_MIN = 0
BOX_MAX = 1
model_num_labels = 2
device = torch.device("cuda:0")

resnet50_model = get_model('resnet50',device)
resnet50_model = load_weights(resnet50_model, args.pretrained_resnet50_path, device)
resnet50_model.eval()

xception_model = get_model('xception',device)
xception_model = load_weights(xception_model, args.pretrained_xception_path, device)
xception_model.eval()

efficient_model = get_model('efficientnet',device)
efficient_model = load_weights(efficient_model, args.pretrained_efficient_path, device)
efficient_model.eval()

resnet50_cam = GradCAM(model=resnet50_model, target_layers=[resnet50_model.layer4[-1]])
xception_cam = GradCAM(model=xception_model, target_layers=[xception_model.conv4])
efficient_cam = GradCAM(model=efficient_model, target_layers=[efficient_model.features[-1]])

cam = [resnet50_cam,xception_cam,efficient_cam]
model_ens = [resnet50_model,xception_model,efficient_model]

test_images_path, test_images_label = read_data(args.data_path)
test_data_set, test_loader = data_preprocess(test_images_path, test_images_label, 224, batch_size)

advGAN = AdvGAN_Attack(device, model_ens, cam,model_num_labels, image_nc, BOX_MIN, BOX_MAX)

current_time = datetime.now()
print(current_time)

advGAN.train(test_loader, epochs)
