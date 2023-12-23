import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# class UNetConvBlock(nn.Module):
#     def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
#         super(UNetConvBlock, self).__init__()
#         self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
#         self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
#         self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
#         self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
#         if use_HIN:
#             self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
#         self.use_HIN = use_HIN

#     def forward(self, x):
#         ## x [1,48,512,512]
#         out = self.conv_1(x) ## [1,48,512,512]
#         if self.use_HIN:
#             out_1, out_2 = torch.chunk(out, 2, dim=1) ## chunk-将一个张量沿着指定的维度进行分割
#             ## out_1 [1,24,512,512] out_2 [1,24,512,512]
#             out = torch.cat([self.norm(out_1), out_2], dim=1) ## [1,48,512,512]
#         out = self.relu_1(out) ## [1,48,512,512]
#         out = self.relu_2(self.conv_2(out)) ## [1,48,512,512]
#         out += self.identity(x) ## [1,48,512,512]
#         return out


# class InvBlock(nn.Module):
#     def __init__(self, channel_num, channel_split_num, clamp=0.8):
#         super(InvBlock, self).__init__()
#         self.split_len1 = channel_split_num
#         self.split_len2 = channel_num - channel_split_num
#         self.clamp = clamp
#         self.F = UNetConvBlock(self.split_len2, self.split_len1)
#         self.G = UNetConvBlock(self.split_len1, self.split_len2)
#         self.H = UNetConvBlock(self.split_len1, self.split_len2)
#         self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

#     def forward(self, x):
#         ## x [1,96,512,512]
#         x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) #将一个序列（如列表或张量）按照指定的索引范围进行切分
#         ## x1 [1,48,512,512] x2 [1,48,512,512]
#         y1 = x1 + self.F(x2) ## [1,48,512,512]
#         self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1) ## [1,48,512,512]
#         y2 = x2.mul(torch.exp(self.s)) + self.G(y1) ## [1,48,512,512]
#         out = torch.cat((y1, y2), 1) ## [1,96,512,512]
#         return out


# class SpaBlock(nn.Module):
#     def __init__(self, nc):
#         super(SpaBlock, self).__init__()
#         self.block = InvBlock(nc,nc//2)

#     def forward(self, x):
#         ## x [1,96,512,512]
#         yy = self.block(x) ## [1,96,512,512]
#         return x+yy


# class FreBlock(nn.Module):
#     def __init__(self, nc):
#         super(FreBlock, self).__init__()
#         self.processmag = nn.Sequential(
#             nn.Conv2d(nc,nc,1,1,0),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(nc,nc,1,1,0))
#         self.processpha = nn.Sequential(
#             nn.Conv2d(nc, nc, 1, 1, 0),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(nc, nc, 1, 1, 0))

#     def forward(self,x):
#         ## x [1,96,512,257]
#         mag = torch.abs(x) ##计算振幅 [1,96,512,257]
#         pha = torch.angle(x) ##计算相位 [1,96,512,257]
#         mag = self.processmag(mag) ##Figure4 FouSpa Block中Fourier branch [1,96,512,257]
#         pha = self.processpha(pha) ##Figure4 FouSpa Block中Fourier branch [1,96,512,257]
#         real = mag * torch.cos(pha) ##计算复数的实部 [1,96,512,257]
#         imag = mag * torch.sin(pha) ##计算复数的虚部 [1,96,512,257]
#         x_out = torch.complex(real, imag) ##创建新的复数张量,实部和虚部分别是变量real和imag的值 [1,96,512,257]
#         return x_out


# class ProcessBlock(nn.Module): ## FouSpa Block
#     def __init__(self, in_nc):
#         super(ProcessBlock,self).__init__()
#         self.spatial_process = SpaBlock(in_nc)
#         self.frequency_process = FreBlock(in_nc)
#         self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
#         self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,1)
#         self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)

#     def forward(self, x):
#         xori = x ## [1,96,512,512]
#         _, _, H, W = x.shape
#         x_freq = torch.fft.rfft2(x, norm='backward') ##计算输入张量x的二维离散傅里叶变换 [1,96,512,257]
#         x = self.spatial_process(x) ## [1,96,512,512]
#         x_freq = self.frequency_process(x_freq) ##经过傅里叶变换，建立了新的复数张量 [1,96,512,257]
#         x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward') ##二维逆傅里叶变换 [1,96,512,512]
#         xcat = torch.cat([x, x_freq_spatial], 1) ## [1,192,512,512]
#         x_out = self.cat(xcat) ## [1,96,512,512]
#         return x_out + xori


# class FreBlockAdjust(nn.Module):
#     def __init__(self, nc):
#         super(FreBlockAdjust, self).__init__()
#         self.processmag = nn.Sequential(
#             nn.Conv2d(nc,nc,1,1,0),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(nc,nc,1,1,0))
#         self.processpha = nn.Sequential(
#             nn.Conv2d(nc, nc, 1, 1, 0),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(nc, nc, 1, 1, 0))
#         self.sft = SFT(nc)
#         self.cat = nn.Conv2d(2*nc,nc,1,1,0)

#     def forward(self, x, y_amp, y_phase):
#         ## x [1,96,512,257] y_amp [1,96,512,257] y_phase [1,96,512,257]
#         mag = torch.abs(x) ## [1,96,512,257]
#         pha = torch.angle(x) ## [1,96,512,257]
#         mag = self.processmag(mag) ## [1,96,512,257]
#         pha = self.processpha(pha) ## [1,96,512,257]
#         mag = self.sft(mag, y_amp) ## [1,96,512,257]
#         pha = self.cat(torch.cat([y_phase, pha], 1)) ## [1,96,512,257]
#         real = mag * torch.cos(pha) ## [1,96,512,257]
#         imag = mag * torch.sin(pha) ## [1,96,512,257]
#         x_out = torch.complex(real, imag) ## [1,96,512,257]
#         return x_out


# class ProcessBlockAdjust(nn.Module): ## Figure4 Adjustment Block
#     def __init__(self, in_nc):
#         super(ProcessBlockAdjust,self).__init__()
#         self.spatial_process = SpaBlock(in_nc)
#         self.frequency_process = FreBlockAdjust(in_nc)
#         self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
#         self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,1)
#         self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)

#     def forward(self, x, y_amp, y_phase):
#         xori = x ## [1,96,512,512]
#         _, _, H, W = x.shape
#         x_freq = torch.fft.rfft2(x, norm='backward') ## [1,96,512,257]
#         x = self.spatial_process(x) ## [1,96,512,512]
#         x_freq = self.frequency_process(x_freq, y_amp, y_phase) ## [1,96,512,257]
#         x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward') ## [1,96,512,512]
#         xcat = torch.cat([x,x_freq_spatial],1) ## [1,192,512,512]
#         x_out = self.cat(xcat) ## [1,192,512,512]
#         return x_out+xori


# class SFT(nn.Module):
#     def __init__(self, nc):
#         super(SFT,self).__init__()
#         self.convmul = nn.Conv2d(nc, nc,3, 1, 1)
#         self.convadd = nn.Conv2d(nc, nc, 3, 1, 1)
#         self.convfuse = nn.Conv2d(2*nc, nc, 1, 1, 0)

#     def forward(self, x, res):
#         ## x [1,96,512,257] res [1,96,512,257]
#         mul = self.convmul(res) ## [1,96,512,257]
#         add = self.convadd(res) ## [1,96,512,257]
#         fuse = self.convfuse(torch.cat([x, mul*x+add],1)) ## [1,96,512,257]
#         return fuse




# class HighNet(nn.Module):
#     def __init__(self, nc):
#         super(HighNet,self).__init__()
#         self.conv0 = nn.PixelUnshuffle(8) ## (*, C, H*r, W*r) to (*, C*r, H, W) 相当于下采样8倍 Xpu [1,192,64,64]
#         self.conv1 = ProcessBlockAdjust(nc*6)
#         self.conv3 = ProcessBlock(nc*12)
#         self.conv4 = ProcessBlock(nc*12)
#         self.conv5 = nn.PixelShuffle(8)
#         self.conv6 = nn.Conv2d(3,96,3,1,1)
#         self.conv7 = nn.Conv2d(96,3,3,1,1)
#         self.convout = nn.Conv2d(nc*12//64, 3, 3, 1, 1)
#         self.trans = nn.Conv2d(6,16,1,1,0)
#         self.con_temp1 = nn.Conv2d(16,16,3,1,1)
#         self.con_temp2 = nn.Conv2d(16,16,3,1,1)
#         self.con_temp3 = nn.Conv2d(16,3,3,1,1)
#         self.LeakyReLU=nn.LeakyReLU(0.1, inplace=False)

#     def forward(self, x, y_down, y_down_amp, y_down_phase):
#         ## x [1,3,512,512] y_down [1,3,512,512] y_down_amp [1,96,512,257] y_down_phase [1,96,512,257]
#         x = self.conv6(x) ## [1,96,512,512]
#         x1 = self.conv1(x, y_down_amp, y_down_phase) ## [1,96,512,512]
#         x1 = self.conv7(x1)
#         xout_temp = self.convout(x1) ## [1,3,512,512]
#         y_aff = self.trans(torch.cat([y_down, xout_temp], 1)) ## [1,16,512,512]
#         xout=self.con_temp3(y_aff) ## [1,3,512,512]
#         return xout




# class LowNet(nn.Module):
#     def __init__(self, in_nc, nc):
#         super(LowNet,self).__init__()
#         self.conv0 = nn.Conv2d(in_nc,nc,1,1,0)
#         self.conv1 = ProcessBlock(nc)
#         self.downsample1 = nn.Conv2d(nc,nc*2,stride=2,kernel_size=2,padding=0)
#         self.conv2 = ProcessBlock(nc*2)
#         self.downsample2 = nn.Conv2d(nc*2,nc*3,stride=2,kernel_size=2,padding=0)
#         self.conv3 = ProcessBlock(nc*3)
#         self.up1 = nn.ConvTranspose2d(nc*5,nc*2,1,1)
#         self.conv4 = ProcessBlock(nc*2)
#         self.up2 = nn.ConvTranspose2d(nc*3,nc*1,1,1)
#         self.conv5 = ProcessBlock(nc)
#         self.convout = nn.Conv2d(nc,nc,1,1,0)
#         self.convoutfinal = nn.Conv2d(nc, 3, 1, 1, 0)
#         self.transamp = nn.Conv2d(nc,nc,1,1,0)
#         self.transpha = nn.Conv2d(nc,nc, 1, 1, 0)

#     def forward(self, x):
#         x = self.conv0(x) ## [1,96,512,512]
#         x01 = self.conv1(x) ## [1,96,512,512]
#         x1 = self.downsample1(x01) ## [1,192,256,256]
#         x12 = self.conv2(x1) ## [1,192,256,256]
#         x2 = self.downsample2(x12) ## [1,288,128,128]
#         x3 = self.conv3(x2) ## [1,288,128,128]
#         x34 = self.up1(torch.cat([F.interpolate(x3,size=(x12.size()[2],x12.size()[3]),mode='bilinear'),x12],1)) ## [1,288,128,128]
#         x4 = self.conv4(x34) ## [1,192,256,256]
#         x4 = self.up2(torch.cat([F.interpolate(x4,size=(x01.size()[2],x01.size()[3]),mode='bilinear'),x01],1)) ## [1,96,512,512]
#         x5 = self.conv5(x4) ## [1,96,512,512]
#         xout = self.convout(x5) ## [1,96,512,512]
#         xout_fre =  torch.fft.rfft2(xout, norm='backward') ## [1,96,512,257]
#         xout_fre_amp, xout_fre_phase = torch.abs(xout_fre), torch.angle(xout_fre) ## xout_fre_amp [1,96,512,257] xout_fre_phase [1,96,512,257]
#         xfinal = self.convoutfinal(xout) ## [1,3,512,512]
#         return xfinal,self.transamp(xout_fre_amp),self.transpha(xout_fre_phase)

#################################################################################################################################################################
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
        
        imnet_in_dim = 256

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
            nn.Conv2d(nc*4, nc*4, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc*4, nc*4, 1, 1, 0))
        
        self.process_amp = nn.Sequential(
            nn.Conv2d(nc*4, nc*4, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc*4, nc*4, 1, 1, 0))
        
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        self.conv_pha = nn.Sequential(
            nn.Conv2d(nc*4, nc*4, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True))

        self.conv_amp = nn.Sequential(
            nn.Conv2d(nc*4, nc*4, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True))
        
        self.Neural = NRN()
        
        self.process_amp_NRN = nn.Sequential(
            nn.Conv2d(3, nc*4, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc*4, nc*4, 1, 1, 0))
        
        self.process_pha_NRN = nn.Sequential(
            nn.Conv2d(3, nc*4, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc*4, nc*4, 1, 1, 0))
        
        self.convoutfinal1 = nn.Conv2d(nc*2, nc, 1, 1, 0)
        self.convoutfinal2 = nn.Conv2d(nc, 3, 1, 1, 0)
        
        self.downsample1 = nn.Conv2d(nc, nc*2, stride=2, kernel_size=2, padding=0)
        self.downsample2 = nn.Conv2d(nc*2, nc*4, stride=2, kernel_size=2, padding=0)
        self.up1 = nn.ConvTranspose2d(nc*8, nc*4, 1, 1)
        self.up2 = nn.ConvTranspose2d(nc*4, nc*2, 1, 1)

    def forward(self, x):
        xori = x ## [1,64,512,512]
        x_down1 = self.downsample1(x)       ## [1,128,256,256]
        x_down2 = self.downsample2(x_down1) ## [1,256,128,128]
        _, _, H, W = x_down2.shape
        
        x_freq = torch.fft.rfft2(x_down2, norm='backward') ##计算输入张量x的二维离散傅里叶变换 [1,256,128,65]
        x_pha = torch.angle(x_freq)       ## [1,256,128,65]
        x_amp = torch.abs(x_freq)         ## [1,256,128,65]
        x_pha = self.process_pha(x_pha)   ## [1,256,128,65]
        x_amp = self.process_amp(x_amp)   ## [1,256,128,65]
        pha_NRN = self.Neural(x_pha)      ## [1,3,128,65]
        pha_NRN = self.process_pha_NRN(pha_NRN) ## [1,256,128,65]
        amp_NRN = self.Neural(x_amp)      ## [1,3,128,65]
        amp_NRN = self.process_amp_NRN(amp_NRN) ## [1,256,128,65]
        x_real = amp_NRN * torch.cos(pha_NRN)   ## [1,256,128,65]
        x_imag = amp_NRN * torch.sin(pha_NRN)   ## [1,256,128,65]
        x_spatial = torch.complex(x_real, x_imag) ## [1,256,128,65]
        x_spatial = torch.fft.irfft2(x_spatial, s=(H, W), norm='backward') ## [1,256,128,128]
        
        y_GAP = self.GAP(x_freq) ## [1,256,1,1]
        y_GAP_pha = torch.angle(y_GAP)  ## [1,256,1,1]
        y_GAP_amp = torch.abs(y_GAP)    ## [1,256,1,1]
        y_GAP = y_GAP.real ## [1,256,1,1]
        y_conv_pha = self.conv_pha(y_GAP) ## [1,256,1,1]
        y_conv_amp = self.conv_amp(y_GAP) ## [1,256,1,1]
        y_pha = y_GAP_pha * y_conv_pha    ## [1,256,1,1]
        y_amp = y_GAP_amp * y_conv_amp    ## [1,256,1,1]
        y_real = y_amp * torch.cos(y_pha) ## [1,256,1,1]
        y_imag = y_amp * torch.sin(y_pha) ## [1,256,1,1]
        y_channel = torch.complex(y_real, y_imag) ## [1,256,1,1]
        y_channel = y_channel.expand_as(x_freq)   ## [1,256,128,65]
        y_channel = torch.fft.irfft2(y_channel, s=(H, W), norm='backward') ## [1,256,128,128]
        
        spa_cha_512 = torch.cat([x_spatial, y_channel], dim=1) ## [1,512,128,128]
        spa_cha_256 = self.up1(F.interpolate(spa_cha_512, size=(x_down1.size()[2], x_down1.size()[3]), mode='bilinear')) ## [1,256,256,256]
        spa_cha_128 = self.up2(F.interpolate(spa_cha_256, size=(x.size()[2], x.size()[3]), mode='bilinear')) ## [1,128,512,512]
        spa_cha_64 = self.convoutfinal1(spa_cha_128) ## [1,64,512,512]
        spa_cha_64 = spa_cha_64 + xori ## [1,64,512,512]
        out = self.convoutfinal2(spa_cha_64)
        return out



class Spatial_Stage(nn.Module):
    def __init__(self, nc): #nc = 64
        super(Spatial_Stage, self).__init__()
        self.DCP = DarkChannel()
        self.PositionalEncoding = PositionalEncoding(num_pos_feats_x=16, num_pos_feats_y=16, num_pos_feats_z=32)
        self.convoutfinal1 = nn.Conv2d(3, nc, 1, 1, 0)
        self.extract1 = nn.Conv2d(3, nc//8, 1, 1, 0)
        self.extract2 = nn.Conv2d(nc//8, nc, 1, 1, 0)
        
        self.process_pha = nn.Sequential(
            nn.Conv2d(nc*4, nc*4, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc*4, nc*4, 1, 1, 0))
        
        self.process_amp = nn.Sequential(
            nn.Conv2d(nc*4, nc*4, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc*4, nc*4, 1, 1, 0))
        
        self.process_spat = nn.Sequential(
            nn.Conv2d(nc*4, nc*4, 1, 1, 0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc*4, nc*4, 1, 1, 0))
        
        self.convoutfinal2 = nn.Conv2d(128, 64, 1, 1, 0)
        self.convoutfinal3 = nn.Conv2d(64, 3, 1, 1, 0)
        self.downsample1 = nn.Conv2d(nc, nc*2, stride=2, kernel_size=2, padding=0)
        self.downsample2 = nn.Conv2d(nc*2, nc*4, stride=2, kernel_size=2, padding=0)
        self.up1 = nn.ConvTranspose2d(nc*4, nc*2, 1, 1)
        self.up2 = nn.ConvTranspose2d(nc*2, nc, 1, 1)
        self.up3 = nn.ConvTranspose2d(nc*4, nc*2, 1, 1)
        self.up4 = nn.ConvTranspose2d(nc*2, nc, 1, 1)

    def forward(self, x):
        xori = x ## [1,3,512,512]
        ones_matrix = torch.ones(1, 3, 512, 512).to(x.device) ## [1,3,512,512]
        x_inverted = torch.sub(ones_matrix, x) ## [1,3,512,512]
        DCP_inverted = self.DCP(x_inverted) ## [1,1,512,512]
        absolute_pos_embed = self.PositionalEncoding(x, DCP_inverted) ## [1,64,512,512]
        x_res = self.convoutfinal1(x)  ## [1,64,512,512]
        absolute_pos_embed = absolute_pos_embed + x_res ## [1,64,512,512]
        
        x_f = self.extract1(x)   ## [1,8,512,512]
        x_f = self.extract2(x_f) ## [1,64,512,512]
        x_down1 = self.downsample1(x_f)     ## [1,128,256,256]
        x_down2 = self.downsample2(x_down1) ## [1,256,128,128]
        _, _, H, W = x_down2.shape
        x_freq = torch.fft.rfft2(x_down2, norm='backward') ## [1,256,128,65]
        x_pha = torch.angle(x_freq)         ## [1,256,128,65]
        x_amp = torch.abs(x_freq)           ## [1,256,128,65]
        x_pha = self.process_pha(x_pha)     ## [1,256,128,65]
        x_amp = self.process_amp(x_amp)     ## [1,256,128,65]
        x_real = x_amp * torch.cos(x_amp)   ## [1,256,128,65]
        x_imag = x_amp * torch.sin(x_amp)   ## [1,256,128,65]
        x_fourier = torch.complex(x_real, x_imag) ## [1,256,128,65]
        x_fourier = torch.fft.irfft2(x_fourier, s=(H, W), norm='backward') ## [1,256,128,128]
        x_fourier = self.up1(F.interpolate(x_fourier, size=(x_down1.size()[2], x_down1.size()[3]), mode='bilinear')) ## [1,128,256,256]
        x_fourier = self.up2(F.interpolate(x_fourier, size=(x_f.size()[2], x_f.size()[3]), mode='bilinear')) ## [1,64,512,512]
        
        x_spatial = self.process_spat(x_down2) ## [1,256,128,128]
        x_spatial = self.up3(F.interpolate(x_spatial, size=(x_down1.size()[2], x_down1.size()[3]), mode='bilinear')) ## [1,128,256,256]
        x_spatial = self.up4(F.interpolate(x_spatial, size=(x_f.size()[2], x_f.size()[3]), mode='bilinear')) ## [1,64,512,512]
        x_spatial = x_spatial * absolute_pos_embed ## [1,64,512,512]
        
        four_spa = torch.cat((x_fourier, x_spatial), 1) ## [1,128,512,512]
        four_spa = self.convoutfinal2(four_spa) ## [1,64,512,512]
        out = self.convoutfinal3(four_spa) ## [1,3,512,512]
        out = out + xori
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
        x_spatial = self.Spatial_Stage(x_frequency) ## [1,3,512,512]
        out = x_spatial + x
        return out