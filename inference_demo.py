import torch.nn as nn
import torchvision.transforms as transforms
from utils.LUT import *
from ipdb import set_trace as S
# import tinycudann as tcnn
from thop import profile
import trilinear as trilinear
import torch.nn.functional as F
import time


class ICELUT(nn.Module): 
    def __init__(self, device, nsw='10+05+10', dim=33, backbone='SRNet'):
        super().__init__()
        self.TrilinearInterpolation = TrilinearInterpolation()
        self.feature = FeatLUT(device)
        self.classifier = torch.from_numpy(np.load('./ICELUT/classifier_int8.npy'))
        self.cluster = torch.split(self.classifier, 64**2, dim=0)
        self.lut_cls = []
        for i in self.cluster:
            self.lut_cls.append(i.unsqueeze(0).to(device))
        self.lut_cat = torch.cat(self.lut_cls,0)
            

        nsw = nsw.split("+")
        num, s, w = int(nsw[0]), int(nsw[1]), int(nsw[2])
        self.CLUTs = CLUT(num, device, dim, s, w)
        self.device = device

    def fuse_basis_to_one(self, msb, lsb, TVMN=None):
        mid_results = self.feature(msb, lsb)
        
        mid_results = ((mid_results*2).long()+32).reshape(5,2)
        index = mid_results[:,0]*64 + mid_results[:,1]
        output = self.lut_cat[torch.arange(5), index.squeeze()]
        output = (output - 32) / 4.0 
        weights = torch.sum(output,0).unsqueeze(0).to(self.device)
        D3LUT = self.CLUTs(weights, TVMN)
        return D3LUT

    def forward(self, img_msb, img_lsb, img_org, TVMN=None):
        D3LUT = self.fuse_basis_to_one(img_msb, img_lsb, TVMN)
        img_res = self.TrilinearInterpolation(D3LUT, img_org)
        return {
            "out": img_res + img_org,
            "3DLUT": D3LUT,
        }

class CLUT(nn.Module):
    def __init__(self, num, device, dim=33, s="-1", w="-1", *args, **kwargs):
        super(CLUT, self).__init__()
        self.num = num
        self.dim = dim
        self.s,self.w = s,w = eval(str(s)), eval(str(w))
        self.s_Layers = nn.Parameter(torch.rand(dim, s)/5-0.1)
        self.w_Layers = nn.Parameter(torch.rand(w, dim*dim)/5-0.1)
        self.LUTs = nn.Parameter(torch.zeros(s*num*3,w))
        basis_lut = np.load('./ICELUT/Basis_lut.npy', allow_pickle=True).item()
        self.s_Layers.data = basis_lut['s']
        self.w_Layers.data = basis_lut['w']
        self.LUTs.data = basis_lut['Basis_LUT']
        self.D3LUTs = self.reconstruct_luts()
        self.D3LUTs = self.D3LUTs.to(device)

    def reconstruct_luts(self):
        dim = self.dim
        num = self.num
        CUBEs = self.s_Layers.mm(self.LUTs.mm(self.w_Layers).reshape(-1,num*3*dim*dim)).reshape(dim,num*3,dim**2).permute(1,0,2).reshape(num,3,self.dim,self.dim,self.dim)
        D3LUTs = cube_to_lut(CUBEs)

        return D3LUTs

    def combine(self, weights, TVMN): # n,num
        dim = self.dim
        num = self.num
        D3LUT = weights.mm(self.D3LUTs.reshape(num,-1)).reshape(-1,3,dim,dim,dim)
        return D3LUT

    def forward(self, weights, TVMN=None):
        lut = self.combine(weights, TVMN)
        return lut


class FeatLUT(torch.nn.Module):
    def __init__(self, device, upscale=4):
        super(FeatLUT, self).__init__()
        self.feature_msb = torch.from_numpy(np.load('./ICELUT/Model_msb_fp32.npy')).to(device)
        self.feature_lsb = torch.from_numpy(np.load('./ICELUT/Model_lsb_fp32.npy')).to(device)
        self.upscale = upscale
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.weights = torch.tensor([16 * 16 * 16, 16 * 16, 16], dtype=torch.float32).to(device).view(1, 3, 1, 1)

    def forward(self, x_in, x_s):
        
        B, C, H, W = x_in.shape
        weights = self.weights

        idx_msb = torch.sum(x_in * weights, dim=1)
        idx_lsb = torch.sum(x_s * weights, dim=1)

        
        idx_msb_flat = idx_msb.view(1, -1).long()
        idx_lsb_flat = idx_lsb.view(1, -1).long()

        out_msb = self.feature_msb[idx_msb_flat,:, :, :].view(1, H, W, -1).permute(0, 3, 1, 2).float()
        out_lsb = self.feature_lsb[idx_lsb_flat,:, :, :].view(1, H, W, -1).permute(0, 3, 1, 2).float()
        
        out = out_lsb + out_msb
        out = self.pool(out)
        temp = torch.round(out * 2) / 2
        out = torch.clamp(temp, -16, 15.5)
        
        return out


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


def test(model, test_dataloader):
    model.eval()
    avg_psnr_out = 0
    for i, batch in enumerate(test_dataloader):
        inputs_msb = batch["input_msb"].to(device)
        inputs_lsb = batch["input_lsb"].to(device)
        inputs_org = batch.get("input_org").to(device)
        targets_org = batch["target_org"].to(device)
        name = os.path.splitext(batch["name"][0])[0]
        results = model(inputs_msb, inputs_lsb, inputs_org, TVMN=None)
        fakes = results["out"]
        
        psnr_out = psnr(fakes, targets_org).item()
        # print(i, psnr_out)
        avg_psnr_out += psnr_out
        if True:
            img_ls = [inputs_org.squeeze().data, fakes.squeeze().data, targets_org.squeeze().data]
            if img_ls[0].shape[0] > 3:
                img_ls = [img.permute(2,0,1) for img in img_ls]
            save_image(img_ls, join('./test_image_output', f"{name}_{psnr_out:.2f}.jpg"), nrow=len(img_ls))
        # break
    # sys.exit()
        
    torch.cuda.empty_cache()
    avg_psnr_out /= len(test_dataloader)

    return avg_psnr_out

if __name__ == "__main__":
    from parameters import *
    from torch.utils.data import DataLoader 
    from utils.LUT import *
    from datasets import *
    from utils.losses import *
    import sys
    from torchvision.utils import save_image
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    model = ICELUT(device).to(device)
    hparams = parser.parse_args()
    hparams.model = ["ICELUT", "10+05+10"]
    test_fivek = False
    test_on_img = True
    test_speed = True

    if test_on_img:
        if test_fivek:
            test_dataloader = DataLoader(
                eval(hparams.dataset)(hparams.data_root, split="test", model='CLUT_quan'),
                batch_size=1,
                shuffle=False,
                num_workers=hparams.num_workers,
            )
            psnr = test(model, test_dataloader)
            print(psnr)

        else:
            inp_dir = './test_image'
            save_path = './test_image_output'
            img_ls = os.listdir(inp_dir)
            os.makedirs(save_path, exist_ok=True)  
            scale = 32
            print(f'Using the inference shape: {scale} X {scale}')
            for img in img_ls:

                ## load the parallel MSB and LSB input
                input_path = os.path.join(inp_dir, img)
                img_input = cv2.cvtColor(cv2.imread(input_path, -1), cv2.COLOR_BGR2RGB)
                img_input_msb = TF.to_tensor((img_input//16)/16)
                img_input_lsb = TF.to_tensor((img_input%16)/16)
                img_input = TF.to_tensor(img_input)

                ## inference using the shape (32, 32)
                img_input_resize_msb, img_input_resize_lsb = TF.resize(img_input_msb, (scale, scale)),TF.resize(img_input_lsb, (scale, scale))
                img_input_resize_msb = torch.round(img_input_resize_msb*16) / 16
                img_input_resize_lsb = torch.round(img_input_resize_lsb*16) / 16
                input_msb = img_input_resize_msb.type(torch.FloatTensor).unsqueeze(0).cuda()
                input_lsb = img_input_resize_lsb.type(torch.FloatTensor).unsqueeze(0).cuda()
                input_org = img_input.type(torch.FloatTensor).unsqueeze(0).cuda()

                ## inference
                output = model(input_msb, input_lsb, input_org)

                ## image saving
                img_ls = [input_org.squeeze().data, output['out'].squeeze().data]
                if img_ls[0].shape[0] > 3:
                    img_ls = [img.permute(2,0,1) for img in img_ls]
                save_image(img_ls, join(save_path, f"{img}.jpg"), nrow=len(img_ls))


    if test_speed:
        scale = 32
        msb = torch.zeros(1, 3, scale, scale).to(device)
        lsb = torch.zeros(1, 3, scale, scale).to(device)

        img_480p = torch.zeros(1, 3, 480, 640).to(device)
        img_720p = torch.zeros(1, 3, 720, 1280).to(device)
        img_2k = torch.zeros(1, 3, 1080, 1920).to(device)
        img_4k = torch.zeros(1, 3, 2160, 4096).to(device)
        ## warm-up hardware
        for i in range(100):
            out = model(msb, lsb, img_480p)

        ## speed test
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        for i in range(500):
            out = model(msb, lsb, img_480p)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        print('inference a 480p image using: ', (t2-t1)/500 * 1000, 'ms')

        ## speed test
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        for i in range(500):
            out = model(msb, lsb, img_720p)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        print('inference a 720p image using: ', (t2-t1)/500 * 1000, 'ms')

            ## speed test
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        for i in range(500):
            out = model(msb, lsb, img_2k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        print('inference a 2k image using: ', (t2-t1)/500 * 1000, 'ms')

            ## speed test
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        for i in range(500):
            out = model(msb, lsb, img_4k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        print('inference a 4k image using: ', (t2-t1)/500 * 1000, 'ms')

        
