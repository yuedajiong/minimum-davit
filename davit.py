import torch

class ConvPosEnc(torch.nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        self.proj = torch.nn.Conv2d(dim,dim, (k,k), (1,1), (k//2,k//2), groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + feat
        return x

class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        else:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            o = x.div(keep_prob) * random_tensor
            return o

class Mlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=torch.nn.GELU):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ChannelBlock(torch.nn.Module):
    class ChannelAttention(torch.nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=False):
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = head_dim ** -0.5
            self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = torch.nn.Linear(dim, dim)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k * self.scale
            attention = k.transpose(-1, -2) @ v
            attention = attention.softmax(dim=-1)
            x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            return x

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop_path=0., act_layer=torch.nn.GELU, norm_layer=torch.nn.LayerNorm, ffn=True):
        super().__init__()
        self.cpe = torch.nn.ModuleList([ConvPosEnc(dim=dim, k=3), ConvPosEnc(dim=dim, k=3)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = self.__class__.ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)

    def forward(self, x_size):
        (x, size) = x_size
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)
        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size

class SpatialBlock(torch.nn.Module):
    class WindowAttention(torch.nn.Module):
        def __init__(self, dim, window_size, num_heads, qkv_bias=True):
            super().__init__()
            self.dim = dim
            self.window_size = window_size
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = head_dim ** -0.5
            self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = torch.nn.Linear(dim, dim)
            self.softmax = torch.nn.Softmax(dim=-1)

        def forward(self, x):
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            return x

    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True, drop_path=0., act_layer=torch.nn.GELU, norm_layer=torch.nn.LayerNorm, ffn=True):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.cpe = torch.nn.ModuleList([ConvPosEnc(dim=dim, k=3), ConvPosEnc(dim=dim, k=3)])
        self.norm1 = norm_layer(dim)
        self.attn = self.__class__.WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)

    def forward(self, x_size):
        def window_partition(x, window_size: int):
            B, H, W, C = x.shape
            x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
            return windows

        def window_reverse(windows, window_size: int, H: int, W: int):
            B = int(windows.shape[0] / (H * W / window_size / window_size))
            x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            return x
        (x, size) = x_size
        H, W = size
        B, L, C = x.shape
        shortcut = self.cpe[0](x, size)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = torch.nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1,self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size

class DualAttentionBlock(torch.nn.ModuleList):
    def __init__(self, block_param, attention_types, embed_dims, num_heads, mlp_ratio, qkv_bias, dpr, layer_offset_id, ffn, window_size):
        super().__init__()
        for layer_id, item in enumerate(block_param):
            module = torch.nn.Sequential()
            for attention_id, attention_type in enumerate(attention_types):
                if attention_type == 'channel':
                    module.append(ChannelBlock(dim=embed_dims[item], num_heads=num_heads[item], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id], norm_layer=torch.nn.LayerNorm, ffn=ffn))
                elif attention_type == 'spatial':
                    module.append(SpatialBlock(dim=embed_dims[item], num_heads=num_heads[item], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id], norm_layer=torch.nn.LayerNorm, ffn=ffn, window_size=window_size))
            self.append(module)

class PatchEmbed(torch.nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=96, overlapped=False):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if self.patch_size[0] == 4:
            self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=(7, 7), stride=self.patch_size, padding=(3, 3))
            self.norm = torch.nn.LayerNorm(embed_dim)
        elif self.patch_size[0] == 2:
            kernel = 3 if overlapped else 2
            pad = 1 if overlapped else 0
            self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=(kernel,kernel), stride=self.patch_size, padding=(pad,pad))
            self.norm = torch.nn.LayerNorm(in_chans)

    def forward(self, x, size):
        H, W = size
        dim = len(x.shape)
        if dim == 3:
            B, HW, C = x.shape
            x = self.norm(x)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = torch.nn.functional.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        newsize = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        if dim == 4:
            x = self.norm(x)
        return x, newsize

import itertools
import math
class DaViT(torch.nn.Module):
    def __init__(self, num_classes, in_chans=3, depths=(1, 1, 3, 1), patch_size=4, embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., qkv_bias=True, drop_path_rate=0.1, norm_layer=torch.nn.LayerNorm, attention_types=('spatial', 'channel'), ffn=True, overlapped_patch=False, weight_init='vit', img_size=224, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        architecture = [[index] * item for index, item in enumerate(depths)]
        self.architecture = architecture   #[[0], [1], [2, 2, 2], [3]]
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_stages = len(self.embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * len(list(itertools.chain(*self.architecture))))]
        self.img_size = img_size
        self.patch_embeds = torch.nn.ModuleList([PatchEmbed(patch_size=patch_size if i == 0 else 2, in_chans=in_chans if i == 0 else self.embed_dims[i - 1], embed_dim=self.embed_dims[i], overlapped=overlapped_patch) for i in range(self.num_stages)])
        main_blocks = []
        for block_id, block_param in enumerate(self.architecture):
            layer_offset_id = len(list(itertools.chain(*self.architecture[:block_id])))
            dual_attention_block =  DualAttentionBlock(block_param, attention_types, embed_dims, num_heads, mlp_ratio, qkv_bias, dpr, layer_offset_id, ffn, window_size)  #torch.nn.ModuleList([torch.nn.Sequential(*[ChannelBlock(dim=self.embed_dims[item], num_heads=self.num_heads[item], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id], norm_layer=torch.nn.LayerNorm, ffn=ffn,) if attention_type == 'channel' else SpatialBlock(dim=self.embed_dims[item], num_heads=self.num_heads[item], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id], norm_layer=torch.nn.LayerNorm, ffn=ffn, window_size=window_size,) if attention_type == 'spatial' else None for attention_id, attention_type in enumerate(attention_types)]) for layer_id, item in enumerate(block_param)])
            main_blocks.append(dual_attention_block)
        self.main_blocks = torch.nn.ModuleList(main_blocks)
        self.norms = norm_layer(self.embed_dims[-1])
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.head = torch.nn.Linear(self.embed_dims[-1], num_classes)

        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            def norm_cdf(x):
                return (1. + math.erf(x / math.sqrt(2.))) / 2.

            if (mean < a - 2 * std) or (mean > b + 2 * std):
                warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " "The distribution of values may be incorrect.", stacklevel=2)
            with torch.no_grad():
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)
                tensor.uniform_(2 * l - 1, 2 * u - 1)
                tensor.erfinv_()
                tensor.mul_(std * math.sqrt(2.))
                tensor.add_(mean)
                tensor.clamp_(min=a, max=b)
                return tensor

        if weight_init == 'conv':
            def _init_conv_weights(m):
                if isinstance(m, torch.nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.normal_(m.weight, std=0.02)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.LayerNorm):
                    torch.nn.init.constant_(m.weight, 1.0)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1.0)
                    torch.nn.init.constant_(m.bias, 0)
            self.apply(_init_conv_weights)
        else:
            def _init_vit_weights(module, name='', head_bias=0., jax_impl=False):
                if isinstance(module, torch.nn.Linear):
                    if name.startswith('head'):
                        torch.nn.init.zeros_(module.weight)
                        torch.nn.init.constant_(module.bias, head_bias)
                    elif name.startswith('pre_logits'):
                        lecun_normal_(module.weight)
                        torch.nn.init.zeros_(module.bias)
                    else:
                        if jax_impl:
                            torch.nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                if 'mlp' in name:
                                    torch.nn.init.normal_(module.bias, std=1e-6)
                                else:
                                    torch.nn.init.zeros_(module.bias)
                        else:
                            trunc_normal_(module.weight, std=.02)
                            if module.bias is not None:
                                torch.nn.init.zeros_(module.bias)
                elif jax_impl and isinstance(module, torch.nn.Conv2d):
                    lecun_normal_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm2d)):
                    torch.nn.init.zeros_(module.bias)
                    torch.nn.init.ones_(module.weight)
            self.apply(_init_vit_weights)

    def forward(self, x):
        x, size = self.patch_embeds[0](x, (x.size(2), x.size(3)))  #(-1, C, H ,W)  ->  [-1, 3136, 96] (56, 56)
        features = [x]
        sizes = [size]
        branches = [0]
        for block_index, block_param in enumerate(self.architecture):   #[[0], [1], [2, 2, 2], [3]]
            branch_ids = sorted(set(block_param))  #[0,1,2,3]
            for branch_id in branch_ids:
                if branch_id not in branches:
                    x, size = self.patch_embeds[branch_id](features[-1], sizes[-1])  #variant
                    features.append(x)
                    sizes.append(size)
                    branches.append(branch_id)
            for layer_index, branch_id in enumerate(block_param):
                features[branch_id], _ = self.main_blocks[block_index][layer_index]((features[branch_id], sizes[branch_id]))
        features[-1] = self.avgpool(features[-1].transpose(1, 2))
        features[-1] = torch.flatten(features[-1], 1)
        x = self.norms(features[-1])
        x = self.head(x)
        return x

def main(image_size=224, num_classes=2, device=("cuda" if torch.cuda.is_available() else "cpu")):
    import torchvision
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize([image_size,image_size]), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=torch.tensor((0.485, 0.456, 0.406)), std=torch.tensor((0.229, 0.224, 0.225)))])
    dataset = torchvision.datasets.ImageFolder('./data/train/', transform)  #.classes:['ficus', 'lego']  .class_to_idx: {'ficus': 0, 'lego': 1}
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=64, num_workers=4, drop_last=True, shuffle=True)  

    network = DaViT(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    for epoch in range(1000):
        for index, (X,Y) in enumerate(dataloader):
            X = X.to(device)
            Y = Y.to(device)
            out = network(X)  #(-1/batch, num_classes)
            O = out.view(-1, out.size(-1))
            loss = torch.nn.functional.cross_entropy(O, Y.view(-1), ignore_index=-1)
            network.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 2.0)
            if 0:
                print('O', torch.max(O.data,1)[1])
                print('Y', Y)
                print()
            if epoch%1==0 and index==0: 
                print('epoch=%04d  loss=%.4f'%(epoch,loss.item()))

if __name__ == '__main__':
    print('Super-AI ...')
    import signal,os; signal.signal(signal.SIGINT, lambda self,code: os._exit(0))
    main()
