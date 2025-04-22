import timm
import torch.nn as nn
import torch
from args import args_parser
args = args_parser()

class ViT_client_side(nn.Module):
    def __init__(self, embedding_layer) -> None:

        super().__init__()
        self.embedding_layer = embedding_layer

    def forward(self, x):
        x = self.embedding_layer(x)
        return x

class ViT_server_side(nn.Module):
    def __init__(self, vit, num_classes) -> None:

        super().__init__()
        self.vit = vit
        self.head_layer = MLP_cls_classes(num_classes= num_classes)

    def forward(self, x):
        x = torch.cat((self.vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        for block_num in range(12):
            x = self.vit.blocks[block_num](x)
        x = self.vit.norm(x)
        cls = x[:, 0, :]
        x = self.head_layer(cls)
        return x

class MLP_cls_classes(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        if args.model_name == 'vit_base_patch32_224':
            self.norm = nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            self.fc = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        elif args.model_name  == 'vit_small_patch32_224':
            self.norm = nn.LayerNorm((384,), eps=1e-06, elementwise_affine=True)
            self.fc = nn.Linear(in_features=384, out_features=num_classes, bias=True)
        self.identity = nn.Identity()
    def forward(self, x):
        x = self.norm(x)
        x = self.identity(x)
        x = self.fc(x)
        return x

class ViT(nn.Module):
    def __init__(
        self, ViT_name, num_classes ,
        in_channels=3, ViT_pretrained = False
        ) -> None:

        super().__init__()

        self.vit = timm.create_model(
            model_name = ViT_name,
            pretrained = ViT_pretrained,
            num_classes = num_classes,
            in_chans = in_channels
        )

        self.head_layer = MLP_cls_classes(num_classes= num_classes)
        self.embedding_layer = self.vit.patch_embed

    def forward(self, x):
        x = self.embedding_layer(x)
        x = torch.cat((self.vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        for block_num in range(12):
            x = self.vit.blocks[block_num](x)
        x = self.vit.norm(x)
        cls = x[:, 0, :]
        x = self.head_layer(cls)
        return x, cls