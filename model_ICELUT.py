import torch.nn as nn
import torchvision.transforms as transforms
from utils.LUT import *
from ipdb import set_trace as S
# import tinycudann as tcnn
from thop import profile
import trilinear as trilinear
import torch.nn.functional as F


class ICELUT(nn.Module): 
    def __init__(self, nsw, dim=33, backbone='PointWise', *args, **kwargs):
        super().__init__()
        print('dump')
        self.TrilinearInterpolation = TrilinearInterpolation()
        print('dump')
        self.pre = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.backbone = eval(backbone)()
        last_channel = self.backbone.last_channel
        self.classifier1 = nn.Sequential(
                nn.Conv2d(last_channel//5, 512,1,1),
                nn.Hardswish(inplace=True),
                # nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(512, int(nsw[:2]),1,1),
        )

        self.classifier2 = nn.Sequential(
                nn.Conv2d(last_channel//5, 512,1,1),
                nn.Hardswish(inplace=True),
                # nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(512, int(nsw[:2]),1,1),
        )
        self.classifier3 = nn.Sequential(
                nn.Conv2d(last_channel//5, 512,1,1),
                nn.Hardswish(inplace=True),
                # nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(512, int(nsw[:2]),1,1),
        )
        self.classifier4 = nn.Sequential(
                nn.Conv2d(last_channel//5, 512,1,1),
                nn.Hardswish(inplace=True),
                # nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(512, int(nsw[:2]),1,1),
        )
        self.classifier5 = nn.Sequential(
                nn.Conv2d(last_channel//5, 512,1,1),
                nn.Hardswish(inplace=True),
                # nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(512, int(nsw[:2]),1,1),
        )
        nsw = nsw.split("+")
        num, s, w = int(nsw[0]), int(nsw[1]), int(nsw[2])
        self.CLUTs = CLUT(num, dim, s, w)

    def fuse_basis_to_one(self, img, img_s, TVMN=None):
        mid_results = self.backbone(img, img_s)
        mid1 = mid_results[:,:2,:,:]
        mid2 = mid_results[:,2:4,:,:]
        mid3 = mid_results[:,4:6,:,:]
        mid4 = mid_results[:,6:8,:,:]
        mid5 = mid_results[:,8:,:,:]

        weights1 = self.classifier1(mid1)[:,:,0,0] # n, num
        weights2 = self.classifier2(mid2)[:,:,0,0] # n, num
        weights3 = self.classifier3(mid3)[:,:,0,0] # n, num
        weights4 = self.classifier4(mid4)[:,:,0,0] # n, num
        weights5 = self.classifier5(mid5)[:,:,0,0] # n, num
        weights = weights1 + weights2 + weights3 + weights4 + weights5
        D3LUT, tvmn_loss = self.CLUTs(weights, TVMN)
        return D3LUT, tvmn_loss    

    def forward(self, img_msb, img_lsb, img_org, TVMN=None):
        D3LUT, tvmn_loss = self.fuse_basis_to_one(img_msb, img_lsb, TVMN)
        img_res = self.TrilinearInterpolation(D3LUT, img_org)
        return {
            "fakes": img_res + img_org,
            "3DLUT": D3LUT,
            "tvmn_loss": tvmn_loss,
        }
    
    def fuse_basis_to_one_test(self, img, img_s, TVMN=None):
        mid_results = self.backbone.forward_test(img, img_s)
        mid1 = mid_results[:,:2,:,:]
        mid2 = mid_results[:,2:4,:,:]
        mid3 = mid_results[:,4:6,:,:]
        mid4 = mid_results[:,6:8,:,:]
        mid5 = mid_results[:,8:,:,:]

        weights1 = self.classifier1(mid1)[:,:,0,0] # n, num
        weights1 = torch.clamp(torch.round(weights1*4)/4, -32, 31.75)
        weights2 = self.classifier2(mid2)[:,:,0,0] # n, num
        weights2 = torch.clamp(torch.round(weights2*4)/4, -32, 31.75)
        weights3 = self.classifier3(mid3)[:,:,0,0] # n, num
        weights3 = torch.clamp(torch.round(weights3*4)/4, -32, 31.75)
        weights4 = self.classifier4(mid4)[:,:,0,0] # n, num
        weights4 = torch.clamp(torch.round(weights4*4)/4, -32, 31.75)
        weights5 = self.classifier5(mid5)[:,:,0,0] # n, num
        weights5 = torch.clamp(torch.round(weights5*4)/4, -32, 31.75)
        weights = weights1 + weights2 + weights3 + weights4 + weights5
        D3LUT, tvmn_loss = self.CLUTs(weights, TVMN)
        return D3LUT, tvmn_loss  

    def forward_test(self, img_msb, img_lsb, img_org, TVMN=None):
        D3LUT, tvmn_loss = self.fuse_basis_to_one_test(img_msb, img_lsb, TVMN)
        img_res = self.TrilinearInterpolation(D3LUT, img_org)
        return {
            "fakes": img_res + img_org,
            "3DLUT": D3LUT,
            "tvmn_loss": tvmn_loss,
        }

class CLUT(nn.Module):
    def __init__(self, num, dim=33, s="-1", w="-1", *args, **kwargs):
        super(CLUT, self).__init__()
        self.num = num
        self.dim = dim
        self.s,self.w = s,w = eval(str(s)), eval(str(w))
        print(dim, self.s, self.w)
        # +: compressed;  -: uncompressed
        if s == -1 and w == -1: # standard 3DLUT
            self.mode = '--'
            self.LUTs = nn.Parameter(torch.zeros(num,3,dim,dim,dim))
        elif s != -1 and w == -1:  
            self.mode = '+-'
            self.s_Layers = nn.Parameter(torch.rand(dim, s)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(s, num*3*dim*dim))
        elif s == -1 and w != -1: 
            self.mode = '-+'
            self.w_Layers = nn.Parameter(torch.rand(w, dim*dim)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(num*3*dim, w))

        else: # full-version CLUT
            self.mode = '++'
            self.s_Layers = nn.Parameter(torch.rand(dim, s)/5-0.1)
            self.w_Layers = nn.Parameter(torch.rand(w, dim*dim)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(s*num*3,w))
        print("n=%d s=%d w=%d"%(num, s, w), self.mode)

    def reconstruct_luts(self):
        dim = self.dim
        num = self.num
        if self.mode == "--":
            D3LUTs = self.LUTs
        else:
            if self.mode == "+-":
                # d,s  x  s,num*3dd  -> d,num*3dd -> d,num*3,dd -> num,3,d,dd -> num,-1
                CUBEs = self.s_Layers.mm(self.LUTs).reshape(dim,num*3,dim*dim).permute(1,0,2).reshape(num,3,self.dim,self.dim,self.dim)
            if self.mode == "-+":
                # num*3d,w x w,dd -> num*3d,dd -> num,3ddd
                CUBEs = self.LUTs.mm(self.w_Layers).reshape(num,3,self.dim,self.dim,self.dim)
            if self.mode == "++":
                # s*num*3, w  x   w, dd -> s*num*3,dd -> s,num*3*dd -> d,num*3*dd -> num,-1
                CUBEs = self.s_Layers.mm(self.LUTs.mm(self.w_Layers).reshape(-1,num*3*dim*dim)).reshape(dim,num*3,dim**2).permute(1,0,2).reshape(num,3,self.dim,self.dim,self.dim)
            D3LUTs = cube_to_lut(CUBEs)

        return D3LUTs

    def combine(self, weights, TVMN): # n,num
        dim = self.dim
        num = self.num

        D3LUTs = self.reconstruct_luts()
        if TVMN is None:
            tvmn_loss = 0
        else:
            tvmn_loss = TVMN(D3LUTs)
        D3LUT = weights.mm(D3LUTs.reshape(num,-1)).reshape(-1,3,dim,dim,dim)
        return D3LUT, tvmn_loss

    def forward(self, weights, TVMN=None):
        lut, tvmn_loss = self.combine(weights, TVMN)
        return lut, tvmn_loss
    

class PointWise(torch.nn.Module):
    def __init__(self, upscale=4):
        super(PointWise, self).__init__()

        self.upscale = upscale

        self.conv1 = nn.Conv2d(3, 32, 1, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(32, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 128, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(128, 256, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(256, 512, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(512, 256, 1, stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(256, 128, 1, stride=1, padding=0, dilation=1)
        self.conv8 = nn.Conv2d(128, 64, 1, stride=1, padding=0, dilation=1)
        self.conv9 = nn.Conv2d(64, 32, 1, stride=1, padding=0, dilation=1)
        self.conv10 = nn.Conv2d(32, 10, 1, stride=1, padding=0, dilation=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv1_s = nn.Conv2d(3, 32, 1, stride=1, padding=0, dilation=1)
        self.conv2_s = nn.Conv2d(32, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3_s = nn.Conv2d(64, 128, 1, stride=1, padding=0, dilation=1)
        self.conv4_s = nn.Conv2d(128, 256, 1, stride=1, padding=0, dilation=1)
        self.conv5_s = nn.Conv2d(256, 512, 1, stride=1, padding=0, dilation=1)
        self.conv6_s = nn.Conv2d(512, 256, 1, stride=1, padding=0, dilation=1)
        self.conv7_s = nn.Conv2d(256, 128, 1, stride=1, padding=0, dilation=1)
        self.conv8_s = nn.Conv2d(128, 64, 1, stride=1, padding=0, dilation=1)
        self.conv9_s = nn.Conv2d(64, 32, 1, stride=1, padding=0, dilation=1)
        self.conv10_s = nn.Conv2d(32, 10, 1, stride=1, padding=0, dilation=1)
        self.last_channel = 10


    def forward(self, x_in, x_s):
        B, C, H, W = x_in.size()

        x = self.conv1(x_in)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.conv7(F.relu(x))
        x = self.conv8(F.relu(x))
        x = self.conv9(F.relu(x))
        x = self.conv10(F.relu(x))
        out_i = x

        x = self.conv1_s(x_s)
        x = self.conv2_s(F.relu(x))
        x = self.conv3_s(F.relu(x))
        x = self.conv4_s(F.relu(x))
        x = self.conv5_s(F.relu(x))
        x = self.conv6_s(F.relu(x))
        x = self.conv7_s(F.relu(x))
        x = self.conv8_s(F.relu(x))
        x = self.conv9_s(F.relu(x))
        x = self.conv10_s(F.relu(x))
        out_s = x
        out = out_s + out_i
        out = self.pool(out)
        # out = torch.clamp(torch.round(out*2)/2, -16, 15.5)

        return out
    
    def forward_test(self, x_in, x_s):
        B, C, H, W = x_in.size()

        x = self.conv1(x_in)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.conv7(F.relu(x))
        x = self.conv8(F.relu(x))
        x = self.conv9(F.relu(x))
        x = self.conv10(F.relu(x))
        out_i = x
        # out_i = torch.clamp(torch.round(out_i*4)/4, -32, 31.75)

        x = self.conv1_s(x_s)
        x = self.conv2_s(F.relu(x))
        x = self.conv3_s(F.relu(x))
        x = self.conv4_s(F.relu(x))
        x = self.conv5_s(F.relu(x))
        x = self.conv6_s(F.relu(x))
        x = self.conv7_s(F.relu(x))
        x = self.conv8_s(F.relu(x))
        x = self.conv9_s(F.relu(x))
        x = self.conv10_s(F.relu(x))
        out_s = x
        # out_s = torch.clamp(torch.round(out_s*4)/4, -32, 31.75)
        out = out_s + out_i
        out = self.pool(out)
        out = torch.clamp(torch.round(out*2)/2, -16, 15.5)
        # print(out)

        return out

    
class TVMN(nn.Module): # (n,)3,d,d,d   or   (n,)3,d
    def __init__(self, dim=33, lambda_smooth=0.0001, lambda_mn=10.0):
        super(TVMN,self).__init__()
        self.dim, self.lambda_smooth, self.lambda_mn = dim, lambda_smooth, lambda_mn
        self.relu = torch.nn.ReLU()
       
        weight_r = torch.ones(1, 1, dim, dim, dim - 1, dtype=torch.float)
        weight_r[..., (0, dim - 2)] *= 2.0
        weight_g = torch.ones(1, 1, dim, dim - 1, dim, dtype=torch.float)
        weight_g[..., (0, dim - 2), :] *= 2.0
        weight_b = torch.ones(1, 1, dim - 1, dim, dim, dtype=torch.float)
        weight_b[..., (0, dim - 2), :, :] *= 2.0        
        self.register_buffer('weight_r', weight_r, persistent=False)
        self.register_buffer('weight_g', weight_g, persistent=False)
        self.register_buffer('weight_b', weight_b, persistent=False)

        self.register_buffer('tvmn_shape', torch.empty(3), persistent=False)


    def forward(self, LUT): 
        dim = self.dim
        tvmn = 0 + self.tvmn_shape
        if len(LUT.shape) > 3: # n,3,d,d,d  or  3,d,d,d
            dif_r = LUT[...,:-1] - LUT[...,1:]
            dif_g = LUT[...,:-1,:] - LUT[...,1:,:]
            dif_b = LUT[...,:-1,:,:] - LUT[...,1:,:,:]
            tvmn[0] =   torch.mean(dif_r**2 * self.weight_r[:,0]) + \
                        torch.mean(dif_g**2 * self.weight_g[:,0]) + \
                        torch.mean(dif_b**2 * self.weight_b[:,0])
            tvmn[1] =   torch.mean(self.relu(dif_r * self.weight_r[:,0])**2) + \
                        torch.mean(self.relu(dif_g * self.weight_g[:,0])**2) + \
                        torch.mean(self.relu(dif_b * self.weight_b[:,0])**2)
            tvmn[2] = 0
        else: # n,3,d  or  3,d
            dif = LUT[...,:-1] - LUT[...,1:]
            tvmn[1] = torch.mean(self.relu(dif))
            dif = dif**2
            dif[...,(0,dim-2)] *= 2.0
            tvmn[0] = torch.mean(dif)
            tvmn[2] = 0

        return self.lambda_smooth*(tvmn[0]+10*tvmn[2]) + self.lambda_mn*tvmn[1]



def CnnActNorm(in_filters, out_filters, kernel_size=3, sp="2_1", normalization=False):
    stride = int(sp.split("_")[0])
    padding = int(sp.split("_")[1])

    layers = [
        nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2),
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))

    return layers

class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        B = x.size(0)
        
        if B == 1:
            assert 1 == trilinear.forward(lut,
                                          x,
                                          output,
                                          dim,
                                          shift,
                                          binsize,
                                          W,
                                          H,
                                          B)
        elif B > 1:
            output = output.permute(1, 0, 2, 3).contiguous()
            assert 1 == trilinear.forward(lut,
                                          x.permute(1,0,2,3).contiguous(),
                                          output,
                                          dim,
                                          shift,
                                          binsize,
                                          W,
                                          H,
                                          B)
            output = output.permute(1, 0, 2, 3).contiguous()

        int_package = torch.IntTensor([dim, shift, W, H, B])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, B = int_package
        dim, shift, W, H, B = int(dim), int(shift), int(W), int(H), int(B)
        binsize = float(float_package[0])

        if B == 1:
            assert 1 == trilinear.backward(x,
                                           x_grad,
                                           lut_grad,
                                           dim,
                                           shift,
                                           binsize,
                                           W,
                                           H,
                                           B)
        elif B > 1:
            assert 1 == trilinear.backward(x.permute(1,0,2,3).contiguous(),
                                           x_grad.permute(1,0,2,3).contiguous(),
                                           lut_grad,
                                           dim,
                                           shift,
                                           binsize,
                                           W,
                                           H,
                                           B)
        return lut_grad, x_grad

# trilinear_need: imgs=nchw, lut=3ddd or 13ddd
class TrilinearInterpolation(torch.nn.Module):
    def __init__(self, mo=False, clip=False):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        
        if lut.shape[0] > 1:
            if lut.shape[0] == x.shape[0]: # n,c,H,W
                use_res = torch.empty_like(x)
                for i in range(lut.shape[0]):
                    use_res[i:i+1] = TrilinearInterpolationFunction.apply(lut[i:i+1], x[i:i+1])[1]
            else:
                n,c,h,w = x.shape
                use_res = torch.empty(n, lut.shape[0], c, h, w).cuda()
                for i in range(lut.shape[0]):
                    use_res[:,i] = TrilinearInterpolationFunction.apply(lut[i:i+1], x)[1]
        else: # n,c,H,W
            use_res = TrilinearInterpolationFunction.apply(lut, x)[1]
        return use_res
        # return torch.clip(TrilinearInterpolationFunction.apply(lut, x)[1],0,1)
