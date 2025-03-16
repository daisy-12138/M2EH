import timm
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers import to_2tuple
import einops


def stem(in_chs, out_chs):

    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )


class Embedding(nn.Module):

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EfficientAdditiveAttention(nn.Module):

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 in_dims=512, token_dim=256, num_heads=2):

        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

        self.linear = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape

        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)  
        key = torch.nn.functional.normalize(key, dim=-1)  

        query_weight = query @ self.w_g  
        A = query_weight * self.scale_factor  

        A = torch.nn.functional.normalize(A, dim=1)  

        G = torch.sum(A * query, dim=1)  

        G = einops.repeat(G, "b d -> b repeat d", repeat=key.shape[1])  

        out = self.Proj(G * key) + query  
        out = self.final(out)  

        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1 * out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            )
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))
        else:
            x = x + self.drop_path(out.reshape(B, H, W, C).permute(0, 3, 1, 2))
            x = x + self.drop_path(self.linear(x))

        return x


class Multiscalefeature(nn.Module):
    def __init__(self, layers=[4, 4, 12, 6], embed_dims=[96, 192, 384, 768],
                 mlp_ratios=4, downsamples=[True, True, True, True],
                 act_layer=nn.GELU,
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 vit_num=1,
                 distillation=True):
        super().__init__()

        self.num_classes = num_classes
        self.patch_embed = stem(3, embed_dims[0])

        block = []
        for i in range(len(layers)):
            for block_idx in range(layers[i]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:i])) / (sum(layers) - 1)

                if layers[i] - block_idx <= vit_num:
                    block.append(EfficientAdditiveAttention(
                        embed_dims[i], mlp_ratio=mlp_ratios,
                        act_layer=act_layer, drop_path=block_dpr,
                        use_layer_scale=use_layer_scale,
                        layer_scale_init_value=layer_scale_init_value))
                else:
                    block.append(ConvBlock(dim=embed_dims[i], hidden_dim=int(mlp_ratios * embed_dims[i]), kernel_size=3))

            if i < len(layers) - 1:
                if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                    block.append(
                        Embedding(
                            patch_size=down_patch_size, stride=down_stride,
                            padding=down_pad,
                            in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                        )
                    )

        self.block = nn.ModuleList(block)
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.dist = distillation
        if self.dist:
            self.dist_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x):
        for block in self.block:
            x = block(x)
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        x = self.head(x.flatten(2).mean(-1))
        return x
    

class AdaptiveFeatureConcatenationMechanism(nn.Module):
    def __init__(self, in_features=3000, hidden_features=500, out_features=3000):
        super(AdaptiveFeatureConcatenationMechanism, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.gate_net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features),
            nn.Sigmoid()
        )

        self.feature_net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x1, x2, x3):

        x = torch.cat((x1, x2, x3), dim=1)
        gates = self.gate_net(x)
        features = self.feature_net(x)
        out = features * gates

        return out


class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels=1, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (
                math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


class M2EH(nn.Module):
    def __init__(self, config, pretrained=True):
        super(M2EH, self).__init__()
        self.branch1 = timm.create_model(config['model']['embedder'], pretrained=pretrained)
        self.branch2 = timm.create_model(config['model']['backbone'], pretrained=pretrained)
        self.branch3 = Multiscalefeature()
        self.adapt = AdaptiveFeatureConcatenationMechanism()
        self.SPP = SPPLayer()
        self.num_features = self.backbone.head.fc.out_features * 3
        self.fc = nn.Linear(self.num_features, self.num_features // 4)
        self.fc2 = nn.Linear(self.num_features // 4, 2)
        self.relu = nn.GELU()
        self.num_features_l = self.backbone.head.fc.out_features * 1
        self.fc_l = nn.Linear(self.num_features_l, self.num_features_l // 4)
        self.fc2_l = nn.Linear(self.num_features_l // 4, 2)

    def forward(self, images):
        x1 = self.branch1(images)
        l1 = self.fc2_l(self.relu(self.fc_l(self.relu(x1))))

        x2 = self.branch2(images)
        l2 = self.fc2_l(self.relu(self.fc_l(self.relu(x2))))

        x3 = self.branch3(images)
        l3 = self.fc2_l(self.relu(self.fc_l(self.relu(x3))))

        x = self.adapt(x1, x2, x3)

        x = x.unsqueeze(2).unsqueeze(3)
        x = self.SPP(x)

        x = self.fc2(self.relu(self.fc(self.relu(x))))

        return x, l1, l2, l3