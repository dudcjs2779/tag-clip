import torch
from torch import nn
from torch.nn import functional as F

class MultiLabelClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text
    
    def get_ground_truth(self, device, sel_gts, gts) -> torch.Tensor:
        
        labels = torch.zeros(len(gts), len(sel_gts), device=device, dtype=torch.long)
        for i, gt in enumerate(gts):
            for j, sel_gt in enumerate(sel_gts):
                if sel_gt.issubset(gt):
                    labels[i, j] = 1

        return labels

    def forward(self, image_features, text_features, logit_scale, sel_gts, gts, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        
        labels = self.get_ground_truth(device, sel_gts, gts)

        total_loss = (
            F.multilabel_soft_margin_loss(logits_per_image, labels) +
            F.multilabel_soft_margin_loss(logits_per_text, labels.T)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
    
class TagClipLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = 0.2

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text
    
    def get_ground_truth(self, device, sel_gts, gts) -> torch.Tensor:
        
        labels = torch.zeros(len(gts), len(gts), device=device, dtype=torch.float16)
        for i, gt in enumerate(gts):
            for j, sel_gt in enumerate(sel_gts):
                intersection_count = len(gt.intersection(sel_gt))
                labels[i, j] = intersection_count / len(sel_gt)

        return labels
    
    def forward(self, image_features, text_features, logit_scale, sel_gts, gts, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        
        labels = self.get_ground_truth(device, sel_gts, gts)

        logits_per_image = F.log_softmax(logits_per_image, dim=1)
        logits_per_text = F.log_softmax(logits_per_text, dim=1)
        labels_T = F.softmax(labels.T**2 / self.temperature, dim=1)
        labels = F.softmax(labels**2 / self.temperature, dim=1)
        total_loss = (
            F.kl_div(logits_per_image, labels) +
            F.kl_div(logits_per_text, labels_T)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
## MultiLabelMarginLoss 