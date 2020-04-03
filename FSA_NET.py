import torch
import torch.nn as nn
import torch.nn.functional as F
from capsule import CapsuleLayer
class  SSRLayer(nn.Module):
    def __init__(self,):
        super(SSRLayer,self).__init__()

    def forward(self,x):
        a = x[0][:, :, 0] * 0
        b = x[0][:, :, 0] * 0
        c = x[0][:, :, 0] * 0

        s1 = 3
        s2 = 3
        s3 = 3
        lambda_d = 1

        di = s1 // 2
        dj = s2 // 2
        dk = s3 // 2

        V = 99

        for i in range(0, s1):
            a = a + (i - di + x[6]) * x[0][:, :, i]
        a = a / (s1 * (1 + lambda_d * x[3]))

        for j in range(0, s2):
            b = b + (j - dj + x[7]) * x[1][:, :, j]
        b = b / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4]))

        for k in range(0, s3):
            c = c + (k - dk + x[8]) * x[2][:, :, k]
        c = c / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4])) / (
            s3 * (1 + lambda_d * x[5]))

        pred = (a + b + c) * V

        return pred

def  MatrixNorm(x): #(-1,c/3,8*8*3)
    # (-1,c/3,64)
    return torch.sum(x,dim = -1,keepdims = True).repeat(1,1,64)

def PrimCaps(x):
    x1,x2,norm = x
    return (x1@x2)/norm

def AggregatedFeatureExtraction(x,num_capsule = 3): # (-1,3,16)
    s1_a = 0
    s1_b = num_capsule//3
    feat_s1_div = x[:,s1_a:s1_b,:] # x[:,0:1,:] (-1,1,16)
    s2_a = num_capsule//3
    s2_b = 2*num_capsule//3
    feat_s2_div = x[:,s2_a:s2_b,:]
    s3_a = 2*num_capsule//3
    s3_b = num_capsule
    feat_s3_div = x[:,s3_a:s3_b,:]

    return feat_s1_div, feat_s2_div, feat_s3_div

class FSANet(nn.Module):
    def __init__(self, S_set):
        super(FSANet,self).__init__()
        self.ssr_layer = SSRLayer()
        self.num_capsule = S_set[0] # 3
        self.dim_capsule = S_set[1] # 16
        self.routings = S_set[2] # 2

        self.num_primcaps = S_set[3] # 8*8*3
        self.m_dim = S_set[4] # 5
        # (-1,c,64) @ (-1,64,16) -> (-1,c,16)
        #self.w1 = nn.Parameter(torch.randn(1,64,16))
        # (-1,16,c) @ (-1,c,3) -> (-1,16,3)
        #self.w2 = nn.Parameter(torch.randn(1,self.num_primcaps,3))
        self.capsule_layer = CapsuleLayer(in_units=self.num_primcaps,in_channels=64,num_units=self.num_capsule,unit_size=self.dim_capsule)
        self.x_layer1 = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(2)
        )
        self.x_layer2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(2)
        )
        self.x_layer3 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(2)
        )
        self.x_layer4 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
        )
        self.s_layer1 = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.s_layer2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.s_layer3 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.s_layer4 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )
        self.x_layer4_ = nn.Sequential(
            nn.Conv2d(128,64,1,1,0),
            nn.ReLU(inplace = True)
        )
        self.x_layer3_ = nn.Sequential(
            nn.Conv2d(64,64,1,1,0),
            nn.ReLU(inplace = True)
        )
        self.x_layer2_ = nn.Sequential(
            nn.Conv2d(32,64,1,1,0),
            nn.ReLU(inplace = True)
        )
        self.s_layer4_ = nn.Sequential(
            nn.Conv2d(128,64,1,1,0),
            nn.Tanh()
        )
        self.s_layer3_ = nn.Sequential(
            nn.Conv2d(64,64,1,1,0),
            nn.Tanh()
        )
        self.s_layer2_ = nn.Sequential(
            nn.Conv2d(32,64,1,1,0),
            nn.Tanh()
        )
        self.agvpool = nn.AvgPool2d(2)
        self.feat_preS1 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.feat_preS2 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.feat_preS3 = nn.Sequential(
            nn.Conv2d(64,1,1,1,0),
            nn.Sigmoid()
        )
        self.sr_matrix1 = nn.Sequential(
            nn.Linear(8*8,self.m_dim * 8*8*3),
            nn.Sigmoid()
        )
        self.sr_matrix2 = nn.Sequential(
            nn.Linear(8*8,self.m_dim * 8*8*3),
            nn.Sigmoid()
        )
        self.sr_matrix3 = nn.Sequential(
            nn.Linear(8*8,self.m_dim * 8*8*3),
            nn.Sigmoid()
        )
        self.SL_matrix = nn.Sequential(
            nn.Linear(8*8*3,int(self.num_primcaps/3)*self.m_dim),
            nn.Sigmoid()
        )
        self.delta_s1 = nn.Sequential(
            nn.Linear(4,3),
            nn.Tanh()
        )
        self.delta_s2 = nn.Sequential(
            nn.Linear(4,3),
            nn.Tanh()
        )
        self.delta_s3 = nn.Sequential(
            nn.Linear(4,3),
            nn.Tanh()
        )
        self.local_s1 = nn.Sequential(
            nn.Linear(4,3),
            nn.Tanh()
        )
        self.local_s2 = nn.Sequential(
            nn.Linear(4,3),
            nn.Tanh()
        )
        self.local_s3 = nn.Sequential(
            nn.Linear(4,3),
            nn.Tanh()
        )
        self.pred_s1 = nn.Sequential(
            nn.Linear(8,9),
            nn.ReLU(inplace = True)
        )
        self.pred_s2 = nn.Sequential(
            nn.Linear(8,9),
            nn.ReLU(inplace = True)
        )
        self.pred_s3 = nn.Sequential(
            nn.Linear(8,9),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        x_layer1 = self.x_layer1(x) # 32
        x_layer2 = self.x_layer2(x_layer1) # 16
        x_layer3 = self.x_layer3(x_layer2) # 8
        x_layer4 = self.x_layer4(x_layer3) # 8

        s_layer1 = self.s_layer1(x)
        s_layer2 = self.s_layer2(s_layer1)
        s_layer3 = self.s_layer3(s_layer2)
        s_layer4 = self.s_layer4(s_layer3)

        s_layer4_ = self.s_layer4_(s_layer4)
        x_layer4_ = self.x_layer4_(x_layer4)
        s_layer3_ = self.s_layer3_(s_layer3)
        x_layer3_ = self.x_layer3_(x_layer3)
        s_layer2_ = self.s_layer2_(s_layer2)
        x_layer2_ = self.x_layer2_(x_layer2)
        feat_s1_pre = s_layer4_ * x_layer4_ # (-1,64,8,8)
        feat_s2_pre = s_layer3_ * x_layer3_ # (-1,64,8,8)
        feat_s3_pre = s_layer2_ * x_layer2_ 
        feat_s3_pre = self.agvpool(feat_s3_pre) #(-1,64,8,8)
        
        feat_preS1 = self.feat_preS1(feat_s1_pre).view(-1,8*8)
        feat_preS2 = self.feat_preS2(feat_s2_pre).view(-1,8*8)
        feat_preS3 = self.feat_preS3(feat_s3_pre).view(-1,8*8)

        sr_matrix1 = self.sr_matrix1(feat_preS1).view(-1,5,8*8*3)
        sr_matrix2 = self.sr_matrix2(feat_preS2).view(-1,5,8*8*3)
        sr_matrix3 = self.sr_matrix3(feat_preS3).view(-1,5,8*8*3)
        
        feat_pre_concat = torch.cat([feat_preS1,feat_preS2,feat_preS3],dim = 1) # (-1,8*8*3)
        SL_matrix = self.SL_matrix(feat_pre_concat).view(-1,int(self.num_primcaps/3),int(self.m_dim)) # (-1,c/3,5)
        S_matrix_s1 = SL_matrix @ sr_matrix1 # (-1,c/3,8*8*3)
        S_matrix_s2 = SL_matrix @ sr_matrix2
        S_matrix_s3 = SL_matrix @ sr_matrix3
        
        norm_S_s1 = MatrixNorm(S_matrix_s1)
        norm_S_s2 = MatrixNorm(S_matrix_s2)
        norm_S_s3 = MatrixNorm(S_matrix_s3)

        feat_s1_pre = feat_s1_pre.view(-1,8*8,64).transpose(1,2).contiguous()
        feat_s2_pre = feat_s2_pre.view(-1,8*8,64).transpose(1,2).contiguous()
        feat_s3_pre = feat_s3_pre.view(-1,8*8,64).transpose(1,2).contiguous()
        feat_pre_concat = torch.cat([feat_s1_pre,feat_s2_pre,feat_s3_pre],dim = 1)
        primcaps_s1 = PrimCaps([S_matrix_s1,feat_pre_concat, norm_S_s1]) # (-1,c/3,8*8*3) @ (-1,64*3,64) -> (-1,c/3,64)
        primcaps_s2 = PrimCaps([S_matrix_s2,feat_pre_concat, norm_S_s2])
        primcaps_s3 = PrimCaps([S_matrix_s3,feat_pre_concat, norm_S_s3])

        primcaps = torch.cat([primcaps_s1,primcaps_s2,primcaps_s3],dim = 1) # (-1,c,64)
        #metric_feat = primcaps @ self.w1
        #metric_feat = (metric_feat.permute(0,2,1) @ self.w2).permute(0,2,1) #(-1,3,16)
        capsule = self.capsule_layer(primcaps)
        feat_s1_div, feat_s2_div, feat_s3_div = AggregatedFeatureExtraction(capsule) # (-1,1,16)
        feat_s1_div = feat_s1_div.view(-1,16)
        feat_s2_div = feat_s2_div.view(-1,16)
        feat_s3_div = feat_s3_div.view(-1,16)
        feat_delta_s1,feat_local_s1,feat_pred_s1 = feat_s1_div[:,0:4],feat_s1_div[:,4:8],feat_s1_div[:,8:16]
        feat_delta_s2,feat_local_s2,feat_pred_s2 = feat_s2_div[:,0:4],feat_s2_div[:,4:8],feat_s2_div[:,8:16]
        feat_delta_s3,feat_local_s3,feat_pred_s3 = feat_s3_div[:,0:4],feat_s3_div[:,4:8],feat_s3_div[:,8:16]
        
        
        delta_s1 = self.delta_s1(feat_delta_s1)
        delta_s2 = self.delta_s2(feat_delta_s2)
        delta_s3 = self.delta_s3(feat_delta_s3)
        
        local_s1 = self.local_s1(feat_local_s1)
        local_s2 = self.local_s2(feat_local_s2)
        local_s3 = self.local_s3(feat_local_s3)

        pred_s1 = self.pred_s1(feat_pred_s1).view(-1,3,3)
        pred_s2 = self.pred_s2(feat_pred_s2).view(-1,3,3)
        pred_s3 = self.pred_s3(feat_pred_s3).view(-1,3,3)

        return self.ssr_layer([pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3])
