
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


MODEL_PATH = '/home/ysd21/ICELUT_camera/FiveK/ICELUT_10+05+10/model0270.pth'

class PointWise(torch.nn.Module):
    def __init__(self):
        super(PointWise, self).__init__()

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
        out = torch.clamp(torch.round(out*2)/2, -16, 15.5)

        return out_i, out_s
    

class SplitFC(nn.Module): 
    def __init__(self, nsw='10+5+10', dim=33, backbone='PointWise', *args, **kwargs):
        super().__init__()

        self.classifier1 = nn.Sequential(
                        nn.Conv2d(2, 512,1,1),
                        nn.Hardswish(inplace=True),
                        # nn.Dropout(p=0.2, inplace=True),
                        nn.Conv2d(512, int(nsw[:2]),1,1),
                )

        self.classifier2 = nn.Sequential(
                nn.Conv2d(2, 512,1,1),
                nn.Hardswish(inplace=True),
                # nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(512, int(nsw[:2]),1,1),
        )
        self.classifier3 = nn.Sequential(
                nn.Conv2d(2, 512,1,1),
                nn.Hardswish(inplace=True),
                # nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(512, int(nsw[:2]),1,1),
        )
        self.classifier4 = nn.Sequential(
                nn.Conv2d(2, 512,1,1),
                nn.Hardswish(inplace=True),
                # nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(512, int(nsw[:2]),1,1),
        )
        self.classifier5 = nn.Sequential(
                nn.Conv2d(2, 512,1,1),
                nn.Hardswish(inplace=True),
                # nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(512, int(nsw[:2]),1,1),
        )
 
        
    def forward(self, mid_results):
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
        weight = torch.cat((weights1, weights2, weights3, weights4, weights5), 0)
        return weight
        

# ========================== Channel LUT ==============================
model_G = PointWise().cuda()
ckpt = model_G.state_dict()
# print(model_G.state_dict().keys())
lm = torch.load('{}'.format(MODEL_PATH))
model2_dict = model_G.state_dict()
state_dict = {k.replace('backbone.', ''):v for k,v in lm.items() if k.replace('backbone.', '') in model2_dict.keys()}
model2_dict.update(state_dict)
model_G.load_state_dict(model2_dict)

# if model_G.keys() not in
model_G.load_state_dict(model_G.state_dict(), strict=True)

### Extract input-output pairs
with torch.no_grad():
    model_G.eval()
    SAMPLING_INTERVAL = 4
    # 1D input
    base = torch.arange(0, 256, 2**SAMPLING_INTERVAL)   # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.cuda().repeat(L)                             # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    third = base.cuda().unsqueeze(1).repeat(1, L*L).reshape(-1) # 256*256*256   0 x65536|1 x65536|...|255 x65536
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)    # [256*256*256, 3]
    input_tensor = onebythree.unsqueeze(1).unsqueeze(1).reshape(-1,3,1,1).float() / 256.0
    out_m, out_l = model_G(input_tensor, input_tensor)
    results_m = out_m.cpu().data.numpy().astype(np.float32)
    results_l = out_l.cpu().data.numpy().astype(np.float32)
    np.save("./ICELUT/Model_lsb_fp32", results_l)
    np.save("./ICELUT/Model_msb_fp32", results_m)



## ======================== Weight LUT ===============================
model_G = SplitFC().cuda()
ckpt = model_G.state_dict()
lm = torch.load('{}'.format(MODEL_PATH))
model2_dict = model_G.state_dict()

state_dict = {k:v for k,v in lm.items() if k in model2_dict.keys()}
model2_dict.update(state_dict)
# print(state_dict['conv1.weight'])
model_G.load_state_dict(model2_dict)
# print(model_G.state_dict()['classifier1.2.bias'])
# print(model_G.state_dict()['conv1.weight'])
# lm = torch.load('{}'.format(MODEL_PATH))
# print(lm['backbone.conv1.weight'])
# print(lm['classifier1.2.bias'])

with torch.no_grad():
    model_G.eval()
    base = torch.arange(-32, 32, 1) / 2.0   # -16 ～ 15.5
    L = base.size(0)
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.cuda().repeat(L)                             # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [64*64, 2]
    input_tensor = onebytwo.unsqueeze(-1).unsqueeze(-1).reshape(-1,2,1,1)
    input_tensor = input_tensor.repeat(1,5,1,1)
    # print(input_tensor)

    out = model_G(input_tensor)
    print(out.shape)
    results_m = out.cpu().data.numpy()
    results_m = (results_m * 4 + 32).astype(np.int8)
    np.save("./ICELUT/classifier_int8", results_m)


ckpt = torch.load('/home/ysd21/ICELUT_camera/FiveK/ICELUT_10+05+10/model0270.pth')
dim, num, s, w = 33, 10, 5, 10
s_Layers = nn.Parameter(torch.rand(dim, s)/5-0.1)
w_Layers = nn.Parameter(torch.rand(w, dim*dim)/5-0.1)
LUTs = nn.Parameter(torch.zeros(s*num*3,w))
s_Layers.data = ckpt['CLUTs.s_Layers']
w_Layers.data = ckpt['CLUTs.w_Layers']
LUTs.data = ckpt['CLUTs.LUTs']
dict = {'s':s_Layers.data,'w':w_Layers.data,'Basis_LUT':LUTs.data}
np.save('./ICELUT/Basis_lut.npy', dict)