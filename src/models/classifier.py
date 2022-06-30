import torch

import src.models.effnet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, "bias") and m.bias is not None:
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, std=0.001)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ClassBlock(torch.nn.Module):
    def __init__(
        self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, classify=True
    ):
        super(ClassBlock, self).__init__()
        self.classify = classify
        add_block = []
        if linear > 0:
            add_block += [torch.nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [torch.nn.BatchNorm1d(linear)]
        if relu:
            add_block += [torch.nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [torch.nn.Dropout(p=droprate)]
        add_block = torch.nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [torch.nn.Linear(linear, class_num)]
        classifier = torch.nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        features = self.add_block(x)
        x = self.classifier(features) if self.classify else -1
        return x, features


class EffNetv2(torch.nn.Module):
    def __init__(
        self,
        class_num=5540,
        features_dim=2048,
        mix_prec=True,
        classify=True,
        imagenet_pretrain=False,
        model_name="s",
    ):
        super().__init__()
        self.mix_prec = mix_prec
        self.backbone = src.models.effnet.EfficientNetV2(
            model_name, n_classes=features_dim, pretrained=imagenet_pretrain
        )
        self.classifier = ClassBlock(1280, class_num, 0.5, linear=features_dim, classify=classify)
        self.EMBEDDING_DIM = features_dim

    def forward(self, x):
        features = self.backbone(x)
        if self.mix_prec:
            features = features.float()

        return self.classifier(features)
