import os
import torch
import torch.nn as nn
from torchvision import transforms
from models.xception import xception as create_xception
from models.resnet import resnet50 as create_resnet
import torch
import os
import torch
from torchvision import transforms
from tools.my_dataset import MyDataSet
from torch_dct import dct_2d,idct_2d
from torchvision import models
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
from models.modelEfficientNet import efficientnet_b0
import models.gramnet as gramnet
from pretrainedmodels import xception
from torchvision import utils



def data_preprocess(images_path, images_label,image_size=224,  batch_size=64,):

    data_transform =transforms.Compose([transforms.Resize(image_size),transforms.ToTensor()])

    data_set = MyDataSet(images_path=images_path,
                         images_class=images_label,
                         transform=data_transform)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    data_loader = torch.utils.data.DataLoader(data_set,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=nw, )
    return data_set, data_loader


def read_data(root: str):

    train_images_path = []  
    train_images_label = []  
    every_class_num = []  
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF"]  

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    image_class = 0 # 0 fake 1 real
    every_class_num.append(len(images))

    for img_path in images:
        train_images_path.append(img_path)
        train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))

    return train_images_path, train_images_label


def load_weights(model, path,device):
    model.load_state_dict(torch.load(path,map_location=device)['model'])
    return model


def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)
            x = module(x)
            if name == self.extracted_layers:
                break
        return x


def DCT(images,if_div=False):

    batch_size, C, H, W = images.shape

    if if_div:
        images = images.view(batch_size, C, -1, 8, 8)
        dct_images = dct_2d(images)/1000
        dct_images = dct_images.reshape(batch_size, C, H, W)
        return dct_images
    else:
        return dct_2d(images)/1000


def IDCT(images_dct):
    return torch.clamp(idct_2d(images_dct * 1000),0,1)

def get_model(name, device):
    
    if name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 2)

    elif name == 'resnet50':
        model = create_resnet(num_classes=2)

    elif name == 'densenet121':
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(1024, 2)

    elif name == 'vgg19':
        model = models.vgg19_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096,2)

    elif name=='efficientnet':
        model = efficientnet_b0(2)

    elif name=='xception':
        model = create_xception(num_classes=2)
    
    elif name=='gramnet':
        model = gramnet.resnet18()

    elif name == 'RFM':
        model = xception(num_classes=2, pretrained=False)

    model = model.to(device)
    
    return model

def adv_save(images, fileName, test_images_path, i):
    for j in range(images.size(0)):
        image = images[j, :, :, :]
        utils.save_image(image, fileName + test_images_path[i + j].split('/')[-1])
