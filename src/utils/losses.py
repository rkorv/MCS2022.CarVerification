import torch
from pytorch_metric_learning import losses
from src.utils.circle_loss import CircleLoss, convert_label_to_similarity


class ReIdentificationLossWithClassification(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.CircleLoss = CircleLoss(m=0.25, gamma=64)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.ContrastLoss = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)

    def forward(self, logits, features, labels):
        BS = labels.shape[0]

        fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features.div(fnorm.expand_as(features))

        ce_loss = self.CrossEntropyLoss(logits, labels)
        contrast_loss = self.ContrastLoss(features, labels)
        circle_loss = self.CircleLoss(*convert_label_to_similarity(features, labels)) / BS

        loss = contrast_loss + ce_loss + circle_loss

        loss_stats = {
            "loss": loss.detach(),
            "CrossEntropyLoss": ce_loss.detach(),
            "ContrastLoss": contrast_loss.detach(),
            "CircleLoss": circle_loss.detach(),
        }

        return loss, loss_stats
