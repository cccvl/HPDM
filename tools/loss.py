import torch
import utils
import torch.nn as nn

loss_mse = torch.nn.MSELoss()

def get_perceptual_loss(vgg, input, adv):
    x_features = vgg(input)
    adv_features = vgg(adv)
    x_gram = [utils.gram(fmap) for fmap in x_features]
    adv_gram = [utils.gram(fmap) for fmap in adv_features]
    style_loss = 0.0
    for j in range(4):
        style_loss += loss_mse(x_gram[j], adv_gram[j])
    style_loss = style_loss

    xcon = x_features[1]
    acon = adv_features[1]
    content_loss = loss_mse(xcon, acon)
    return style_loss, content_loss

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.epsilon = 1e-8

    def forward(self, map_pred,
                map_gtd):
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()

        map_pred = map_pred.view(1, -1) 
        map_gtd = map_gtd.view(1, -1) 

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon)  

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon)  
        map_pred = map_pred / (
                    torch.sum(map_pred) + self.epsilon)  
        map_gtd = map_gtd / (
                    torch.sum(map_gtd) + self.epsilon)  

        KL = torch.log(map_gtd / (map_pred + self.epsilon) + self.epsilon)
        KL = map_gtd * KL
        KL = torch.sum(KL)

        return KL





















