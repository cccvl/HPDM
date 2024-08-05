import torch.nn as nn
import torch
import model as models
import torch.nn.functional as F
import os
from tools.cam_module import get_cam_mask
from tools.loss import get_perceptual_loss as vggloss
from tools.loss import KLLoss
from tools.vgg import Vgg16
from utils import DCT,IDCT
from pytorch_msssim import ssim
from tqdm import tqdm
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))

models_path = '{}/baseline/models/'.format(parent_dir)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,device,model_ens,cam,model_num_labels,image_nc,box_min,box_max):
        self.device = device
        self.model_num_labels = model_num_labels
        self.model1, self.model2, self.model3 = model_ens
        self.resnet_cam, self.efficient_cam, self.xception_cam = cam
        self.input_nc = image_nc
        self.output_nc = image_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)
        self.netG_fre = models.Generator(self.gen_input_nc, image_nc).to(device)

        self.bceloss = nn.BCEWithLogitsLoss()
        self.klloss = KLLoss()
        self.vgg = Vgg16().to(device)
        self.ssim = ssim

        self.netG.apply(weights_init)
        self.netG_fre.apply(weights_init)
        self.netDisc.apply(weights_init)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=0.001)
        self.optimizer_G_fre = torch.optim.Adam(self.netG_fre.parameters(),lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):

        for i in range(1):

            # SPACE
            perturbation = self.netG(x)
            cam_soft_masks, cam_hard_masks = get_cam_mask(x, self.resnet_cam, self.xception_cam, self.efficient_cam)
            perturbation = torch.mul(perturbation, cam_hard_masks.to(self.device))
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            # FRE
            dct_x = DCT(x)
            per_fre = self.netG_fre(dct_x)
            adv_images_fre = IDCT(torch.clamp(per_fre, -dct_x.max()*0.1, dct_x.max()*0.1) + dct_x)
            perturbation_fre = torch.mul(adv_images_fre - x, cam_hard_masks.to(self.device))
            adv_images_fre = perturbation_fre + x
            adv_images_fre = torch.clamp(adv_images_fre, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            pred_fake_fre = self.netDisc(adv_images_fre.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device)) + F.mse_loss(pred_fake_fre, torch.zeros_like(pred_fake_fre, device=self.device))
            loss_D_fake.backward()

            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        for i in range(1):
            
            self.optimizer_G.zero_grad()
            self.optimizer_G_fre.zero_grad()

            pred_fake = self.netDisc(adv_images)
            pred_fake_fre = self.netDisc(adv_images_fre)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device)) + F.mse_loss(pred_fake_fre, torch.ones_like(pred_fake_fre, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            loss_perturb = torch.mean(torch.norm(torch.mul(perturbation, (1-cam_soft_masks).to(self.device)).view(perturbation.shape[0], -1), 2, dim=1))
            loss_perturb_fre = torch.mean(torch.norm(torch.mul(perturbation_fre, (1-cam_soft_masks).to(self.device)).reshape(perturbation_fre.shape[0], -1), 2, dim=1))

            logits_resnet = self.model1(adv_images)
            logits_xception = self.model2(adv_images)
            logits_efficient = self.model3(adv_images)
            logits_resnet_fre = self.model1(adv_images_fre)
            logits_xception_fre = self.model2(adv_images_fre)
            logits_efficient_fre = self.model3(adv_images_fre)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[torch.ones_like(labels, device=self.device)]
            loss_adv = (self.bceloss(logits_resnet, onehot_labels) + self.bceloss(logits_xception, onehot_labels) + self.bceloss(logits_efficient, onehot_labels))/3
            loss_adv_fre = (self.bceloss(logits_resnet_fre, onehot_labels) + self.bceloss(logits_xception_fre, onehot_labels) + self.bceloss(logits_efficient_fre, onehot_labels))/3

            STYLE_WEIGHT = 1e5
            CONTENT_WEIGHT = 1e0
            loss_style, loss_content = vggloss(self.vgg, x, adv_images)
            loss_style_fre, loss_content_fre = vggloss(self.vgg, x, adv_images_fre)
            loss_vgg = STYLE_WEIGHT*loss_style + CONTENT_WEIGHT*loss_content
            loss_vgg_fre = STYLE_WEIGHT*loss_style_fre + CONTENT_WEIGHT*loss_content_fre

            features_resnet_x = self.model1.getfeatures(x)
            features_resnet_adv = self.model1.getfeatures(adv_images)
            features_xception_x = self.model2.getfeatures(x)
            features_xception_adv = self.model2.getfeatures(adv_images)
            features_efficient_x = self.model3.getfeatures(x)
            features_efficient_adv = self.model3.getfeatures(adv_images)
            features_resnet_adv_fre = self.model1.getfeatures(adv_images_fre)
            features_xception_adv_fre = self.model2.getfeatures(adv_images_fre)
            features_efficient_adv_fre = self.model3.getfeatures(adv_images_fre)
            loss_latent = 3 / (self.klloss(features_resnet_adv, features_resnet_x) + self.klloss(features_xception_adv, features_xception_x) + self.klloss(features_efficient_adv, features_efficient_x))
            loss_latent_fre = 3 / (self.klloss(features_resnet_adv_fre, features_resnet_x) + self.klloss(features_xception_adv_fre, features_xception_x) + self.klloss(features_efficient_adv_fre, features_efficient_x))

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb + loss_vgg + loss_latent
            loss_G_fre = adv_lambda * loss_adv_fre + pert_lambda * loss_perturb_fre + loss_vgg_fre + loss_latent_fre
            loss_ssim = 1 - self.ssim(adv_images*255.0, adv_images_fre*255.0, data_range=255, size_average=True)
            loss_all = loss_G + loss_G_fre + 10 * loss_ssim
            loss_all.backward()

            self.optimizer_G.step()
            self.optimizer_G_fre.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item(), loss_vgg.item(), loss_latent.item(), loss_perturb_fre.item(), loss_adv_fre.item(), loss_vgg_fre.item(), loss_latent_fre.item(), loss_ssim.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            loss_vgg_sum = 0
            loss_latent_sum = 0
            loss_perturb_sum_fre = 0
            loss_adv_sum_fre = 0
            loss_vgg_sum_fre = 0
            loss_latent_sum_fre = 0
            loss_ssim_sum = 0

            train_dataloader = tqdm(train_dataloader)

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, loss_vgg_batch, loss_latent_batch, \
                    loss_perturb_fre_batch, loss_adv_fre_batch, loss_vgg_fre_batch, loss_latent_fre_batch, loss_ssim_batch = self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_ssim_sum += loss_ssim_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                loss_vgg_sum += loss_vgg_batch
                loss_latent_sum += loss_latent_batch
                loss_perturb_sum_fre += loss_perturb_fre_batch
                loss_adv_sum_fre += loss_adv_fre_batch
                loss_vgg_sum_fre += loss_vgg_fre_batch
                loss_latent_sum_fre += loss_latent_fre_batch

            num_batch = len(train_dataloader)
            print("epoch %d:\n\
                  loss_D: %.3f, loss_G_fake: %.3f, loss_ssim: %.4f\n\
                  loss_perturb: %.3f, loss_adv: %.3f, loss_vgg: %.3f, loss_latent: %.3f\n\
                  loss_perturb_fre: %.3f, loss_adv_fre: %.3f, loss_vgg_fre: %.3f, loss_latent_fre: %.3f\n" 
                  %(epoch, 
                   loss_D_sum/num_batch, loss_G_fake_sum/num_batch,loss_ssim_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch, loss_vgg_sum/num_batch, loss_latent_sum/num_batch,
                   loss_perturb_sum_fre/num_batch, loss_adv_sum_fre/num_batch, loss_vgg_sum_fre/num_batch, loss_latent_sum_fre/num_batch))

            netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
            torch.save(self.netG.state_dict(), netG_file_name)
