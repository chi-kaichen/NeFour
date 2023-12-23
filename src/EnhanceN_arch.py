import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

L = 8
hidden_list = [256,256,256]

def make_coord(shape, ranges=None, flatten=True): ## 生成坐标网格
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)



class NRN(nn.Module):
    def __init__(self, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        imnet_in_dim = 64

        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2+4*L
        if self.cell_decode:
            imnet_in_dim += 2

        self.imnet = MLP(imnet_in_dim, 3, hidden_list)

    def positional_encoding(self, input, L):
        shape = input.shape ## [1,131584,2]
        freq = 2**torch.arange(L, dtype=torch.float32).cuda()*np.pi
        spectrum = input[..., None] * freq ## [1,131584,2,8]
        sin, cos = spectrum.sin(), spectrum.cos() ## sin cos:[1,131584,2,8]
        input_enc = torch.stack([sin,cos],dim=-2) ## [1,131584,2,2,8]
        input_enc = input_enc.view(*shape[:-1],-1) ## [1,131584,32]
        return input_enc

    def query_rgb(self, inp, coord, cell=None):
        feat = inp ## [1, 64, 512, 257]
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            ## [1, 576, 512, 257]
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:]) ## [1,2,512,257]

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                bs, q, h, w = feat.shape ## bs:1 q:576 h:512 w:257
                q_feat = feat.view(bs, q, -1).permute(0, 2, 1)        ## [1,131584,576]

                bs, q, h, w = feat_coord.shape
                q_coord = feat_coord.view(bs, q, -1).permute(0, 2, 1) ## [1,131584,2]

                points_enc = self.positional_encoding(q_coord, L=L)   ## [1,131584,32]
                q_coord = torch.cat([q_coord, points_enc], dim=-1)    ## [1,131584,34]

                rel_coord = coord - q_coord                           ## [1,131584,34]
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)          ## [1,131584,610]

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)          ## [1,131584,612]

                bs, q = coord.shape[:2]  ## bs:1 q:131584
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1) ## [1,131584,3]
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1]) ## [1,131584]
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0) ##[1, 131584]

        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0

        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1) ## [1,131584,3]
            
        bs, q, h, w = feat.shape ## bs:1 q:576 h:512 w:257
        ret = ret.view(bs, h, w, -1).permute(0, 3, 1, 2) ## [1,3,512,257]
        return ret

    def forward(self, inp):
        h, w = inp.shape[2], inp.shape[3] ## h:512 w:257
        coord = make_coord((h, w)).cuda() ## [131584,2]
        cell = torch.ones_like(coord)     ## [131584,2]
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell = cell.unsqueeze(0)   ## [1,131584,2]
        coord = coord.unsqueeze(0) ## [1,131584,2]
        points_enc = self.positional_encoding(coord, L = L) ## [1,131584,32]
        coord = torch.cat([coord, points_enc], dim=-1)      ## [1,131584,34]
        out = self.query_rgb(inp, coord, cell) ## [1,3,512,257]
        return out



class DarkChannel(nn.Module):
    def __init__(self, kernel_size = 15):
        super(DarkChannel, self).__init__()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)
    def forward(self, x):
        H, W = x.size()[2], x.size()[3]
        # maximum among three channels
        x, _ = x.min(dim=1, keepdim=True)
        x = nn.ReflectionPad2d(self.pad_size)(x)
        x = self.unfold(x)
        x = x.unsqueeze(1)
        # maximum in (k, k) patch
        dark_map, _ = x.min(dim=2, keepdim=False)
        x = dark_map.view(-1, 1, H, W)
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, num_pos_feats_x=16, num_pos_feats_y=16, num_pos_feats_z=32, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats_x = num_pos_feats_x
        self.num_pos_feats_y = num_pos_feats_y
        self.num_pos_feats_z = num_pos_feats_z
        self.num_pos_feats = max(num_pos_feats_x, num_pos_feats_y, num_pos_feats_z)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, depth):
        b, c, h, w = x.size()
        b_d, c_d, h_d, w_d = depth.size()
        assert b == b_d and c_d == 1 and h == h_d and w == w_d
        
        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            y_embed = torch.arange(h, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(b, 1, w) ## [1,512,512]
            x_embed = torch.arange(w, dtype=torch.float32, device=x.device).repeat(b, h, 1) ## [1,512,512]
        z_embed = depth.squeeze().to(dtype=torch.float32, device=x.device) ## [512,512]

        if self.normalize:
            eps = 1e-6
            if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
                y_embed = y_embed / (y_embed.max() + eps) * self.scale ## [1,512,512]
                x_embed = x_embed / (x_embed.max() + eps) * self.scale ## [1,512,512]
            z_embed_max, _ = z_embed.reshape(b, -1).max(1)
            z_embed = z_embed / (z_embed_max[:, None, None] + eps) * self.scale ## [1,512,512]

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) ##[32]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)            ##[32]

        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            pos_x = x_embed[:, :, :, None] / dim_t[:self.num_pos_feats_x] ## [1,512,512,32]
            pos_y = y_embed[:, :, :, None] / dim_t[:self.num_pos_feats_y] ## [1,512,512,32]
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) ## [1,512,512,32]
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3) ## [1,512,512,32]

        pos_z = z_embed[:, :, :, None] / dim_t[:self.num_pos_feats_z] ## [1,512,512,32]
        pos_z = torch.stack((pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()), dim=4).flatten(3) ## [1,512,512,32]

        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            pos = torch.cat((pos_x, pos_y, pos_z), dim=3).permute(0, 3, 1, 2) ## [1,96,512,512]
        else:
            pos = pos_z.permute(0, 3, 1, 2)
        return pos



class Frequency_Stage(nn.Module):
    def __init__(self, nc): #nc = 64
        super(Frequency_Stage, self).__init__()
        self.process_pha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        
        self.process_amp = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        self.conv_pha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True))

        self.conv_amp = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True))
        
        self.Neural = NRN()
        
        self.process_amp_NRN = nn.Sequential(
            nn.Conv2d(3, nc, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        
        self.process_pha_NRN = nn.Sequential(
            nn.Conv2d(3, nc, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        
        self.convoutfinal1 = nn.Conv2d(nc*2, nc, 1, 1, 0)
        self.convoutfinal2 = nn.Conv2d(nc, 3, 1, 1, 0)

    def forward(self, x):
        xori = x ## [1,64,512,512]
        _, _, H, W = x.shape
        
        x_freq = torch.fft.rfft2(x, norm='backward') ##计算输入张量x的二维离散傅里叶变换 [1,64,512,257]
        x_pha = torch.angle(x_freq)       ## [1,64,512,257]
        x_amp = torch.abs(x_freq)         ## [1,64,512,257]
        x_pha = self.process_pha(x_pha)   ## [1,64,512,257]
        x_amp = self.process_amp(x_amp)   ## [1,64,512,257]
        pha_NRN = self.Neural(x_pha)      ## [1,3,512,257]
        pha_NRN = self.process_pha_NRN(pha_NRN) ## [1,64,512,257]
        amp_NRN = self.Neural(x_amp)      ## [1,3,512,257]
        amp_NRN = self.process_amp_NRN(amp_NRN) ## [1,64,512,257]
        x_real = amp_NRN * torch.cos(pha_NRN)   ## [1,64,512,257]
        x_imag = amp_NRN * torch.sin(pha_NRN)   ## [1,64,512,257]
        x_spatial = torch.complex(x_real, x_imag) ## [1,64,512,257]
        x_spatial = torch.fft.irfft2(x_spatial, s=(H, W), norm='backward') ## [1,64,512,512]
        
        y_GAP = self.GAP(x_freq) ## [1,64,1,1]
        y_GAP_pha = torch.angle(y_GAP)  ## [1,64,1,1]
        y_GAP_amp = torch.abs(y_GAP)    ## [1,64,1,1]
        y_GAP = y_GAP.real ## [1,64,1,1]
        y_conv_pha = self.conv_pha(y_GAP) ## [1,64,1,1]
        y_conv_amp = self.conv_amp(y_GAP) ## [1,64,1,1]
        y_pha = y_GAP_pha * y_conv_pha    ## [1,64,1,1]
        y_amp = y_GAP_amp * y_conv_amp    ## [1,64,1,1]
        y_real = y_amp * torch.cos(y_pha) ## [1,64,1,1]
        y_imag = y_amp * torch.sin(y_pha) ## [1,64,1,1]
        y_channel = torch.complex(y_real, y_imag) ## [1,64,1,1]
        y_channel = y_channel.expand_as(x_freq)   ## [1,64,512,257]
        y_channel = torch.fft.irfft2(y_channel, s=(H, W), norm='backward') ## [1,64,512,512]
        
        spa_cha = torch.cat([x_spatial, y_channel], dim=1) ## [1,128,512,512]
        spa_cha = self.convoutfinal1(spa_cha) ## [1,64,512,512]
        spa_cha = spa_cha + xori ## [1,64,512,512]
        out = self.convoutfinal2(spa_cha) ## [1,3,512,512]
        return out



class Spatial_Stage(nn.Module):
    def __init__(self, nc): #nc = 64
        super(Spatial_Stage, self).__init__()
        self.DCP = DarkChannel()
        self.PositionalEncoding = PositionalEncoding(num_pos_feats_x=16, num_pos_feats_y=16, num_pos_feats_z=32)
        self.convoutfinal1 = nn.Conv2d(3, nc, 1, 1, 0)
        
        self.process_pha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        
        self.process_amp = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        
        self.process_spat = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        
        self.convoutfinal2 = nn.Conv2d(3, nc, 1, 1, 0)
        self.convoutfinal3 = nn.Conv2d(nc*2, nc, 1, 1, 0)
        self.convoutfinal4 = nn.Conv2d(nc, 3, 1, 1, 0)

    def forward(self, x):
        xori = x ## [1,3,512,512]
        _, _, H, W = x.shape
        ones_matrix = torch.ones(1, 3, 512, 512).to(x.device) ## [1,3,512,512]
        x_inverted = torch.sub(ones_matrix, x) ## [1,3,512,512]
        DCP_inverted = self.DCP(x_inverted) ## [1,1,512,512]
        absolute_pos_embed = self.PositionalEncoding(x, DCP_inverted) ## [1,64,512,512]
        x_res = self.convoutfinal1(x)  ## [1,64,512,512]
        absolute_pos_embed = absolute_pos_embed + x_res ## [1,64,512,512]
        
        x_f = self.convoutfinal2(x) ## [1,64,512,512]
        x_freq = torch.fft.rfft2(x_f, norm='backward') ## [1,64,512,257]
        x_pha = torch.angle(x_freq)         ## [1,64,512,257]
        x_amp = torch.abs(x_freq)           ## [1,64,512,257]
        x_pha = self.process_pha(x_pha)     ## [1,64,512,257]
        x_amp = self.process_amp(x_amp)     ## [1,64,512,257]
        x_real = x_amp * torch.cos(x_amp)   ## [1,64,512,257]
        x_imag = x_amp * torch.sin(x_amp)   ## [1,64,512,257]
        x_fourier = torch.complex(x_real, x_imag) ## [1,64,512,257]
        x_fourier = torch.fft.irfft2(x_fourier, s=(H, W), norm='backward') ## [1,64,512,512]
        x_spatial = self.process_spat(x_f) ## [1,64,512,512]
        x_spatial = x_spatial * absolute_pos_embed ## [1,64,512,512]

        four_spa = torch.cat((x_fourier, x_spatial), 1) ## [1,128,512,512]
        four_spa = self.convoutfinal3(four_spa) ## [1,64,512,512]
        four_spa = self.convoutfinal4(four_spa) ## [1,3,512,512]
        out = four_spa + xori
        return out





class InteractNet(nn.Module):
    def __init__(self, nc=16):
        super(InteractNet,self).__init__()
        self.extract1 =  nn.Conv2d(3, nc//2, 1, 1, 0)
        self.extract2 =  nn.Conv2d(nc//2, nc*4, 1, 1, 0)
        self.Frequency_Stage = Frequency_Stage(nc*4)
        self.Spatial_Stage = Spatial_Stage(nc*4)

    def forward(self, x):
        ## x [1,3,512,512]
        x_f = self.extract1(x)   ## [1,8,512,512]
        x_f = self.extract2(x_f) ## [1,64,512,512]
        x_frequency = self.Frequency_Stage(x_f) ## [1,3,512,512]
        out = x_frequency + x
        return out