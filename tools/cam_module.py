from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch


def get_cam_mask(input_tensor, resnetcam, xceptioncam, efficientcam):

        # fake
        target_category = []
        for i in range(len(input_tensor)):
             target_category.append(ClassifierOutputTarget(0))

        resnet_cam = resnetcam(input_tensor=input_tensor,targets=target_category)
        efficient_cam = efficientcam(input_tensor=input_tensor, targets=target_category)
        xception_cam = xceptioncam(input_tensor=input_tensor, targets=target_category)
        
        resnet_cam = torch.from_numpy(resnet_cam)
        efficient_cam = torch.from_numpy(efficient_cam)
        xception_cam=torch.from_numpy(xception_cam)

        grayscale_cam = (resnet_cam+efficient_cam+xception_cam)/3.0

        # real
        target_category1 = []
        for i in range(len(input_tensor)):
            target_category1.append(ClassifierOutputTarget(1))

        resnet_cam1 = resnetcam(input_tensor=input_tensor,targets=target_category1)
        efficient_cam1 = efficientcam(input_tensor=input_tensor, targets=target_category1)
        xception_cam1 = xceptioncam(input_tensor=input_tensor, targets=target_category1)

        resnet_cam1 = torch.from_numpy(resnet_cam1)
        efficient_cam1 = torch.from_numpy(efficient_cam1)
        xception_cam1 = torch.from_numpy(xception_cam1)

        grayscale_cam1 = (resnet_cam1+efficient_cam1+xception_cam1)/3.0
        grayscale_cam = torch.clamp(grayscale_cam + grayscale_cam1, 0, 1)

        grayscale_cam = grayscale_cam.unsqueeze(1)
        cam_mask = grayscale_cam.expand(input_tensor.size())
        cam_soft_masks = torch.where(cam_mask < 0.3, torch.zeros(cam_mask.size()), cam_mask)
        cam_hard_masks = torch.where(cam_soft_masks >= 0.3, torch.ones(cam_mask.size()), cam_soft_masks)

        return cam_soft_masks, cam_hard_masks
        


















