import torch
import torch.nn as nn
import torch.nn.functional as F
def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)
class BaseBlock(nn.Module):
    def __init__(self,in_channels,out_channels,activation = "relu"):
        super(BaseBlock,self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size = 3,stride = 1,padding=1),
            nn.BatchNorm2d(out_channels),
        )        
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
    def forward(self,x):
        x = self.seq(x)
        return self.activation(x)
class FeatureExtrator(nn.Module):
    def __init__(self,in_channels,out_channels=64,pool_size=1):
        super(FeatureExtrator,self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.AvgPool2d(pool_size)
        )
    def forward(self,x):
        return self.seq(x)

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,  # fixme constant
                               kernel_size=9,  # fixme constant
                               stride=2, # fixme constant
                               bias=True)

    def forward(self, x):
        return self.conv0(x)

class CapsuleLayer(nn.Module):
    def __init__(self, in_units = 21, in_channels = 64, num_units = 3, unit_size = 16, use_routing = True):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing

        if self.use_routing:
            # In the paper, the deeper capsule layer(s) with capsule inputs (DigitCaps) use a special routing algorithm
            # that uses this weight matrix.
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
        else:
            # The first convolutional capsule layer (PrimaryCapsules in the paper) does not perform routing.
            # Instead, it is composed of several convolutional units, each of which sees the full input.
            # It is implemented as a normal convolutional layer with a special nonlinearity (squash()).
            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_channels)
                self.add_module("unit_" + str(unit_idx), unit)
                return unit
            self.units = [create_conv_unit(i) for i in range(self.num_units)]

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, height, width).
        u = [self.units[i](x) for i in range(self.num_units)]

        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1)

        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.num_units, -1)

        # Return squashed outputs.
        return CapsuleLayer.squash(u)

    def routing(self, x):
        batch_size = x.size(0)

        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)

        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)

        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)

        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x)

        # Initialize routing logits to zero.
        b_ij = torch.zeros(1, self.in_channels, self.num_units, 1)

        # Iterative routing.
        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)

            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)
class SSRModule(nn.Module):
    def __init__(self,):
        super(SSRModule,self).__init__()
        self.en1 = nn.Sequential(nn.Linear(16,3),nn.Tanh()) # en
        self.p1 = nn.Sequential(nn.Linear(16,3),nn.ReLU()) # p,每个stage的类别的概率
        self.delta1 = nn.Sequential(nn.Linear(16,1),nn.Tanh()) # delta
        self.en2 = nn.Sequential(nn.Linear(16,3),nn.Tanh()) # en
        self.p2 = nn.Sequential(nn.Linear(16,3),nn.ReLU()) # p,每个stage的类别的概率
        self.delta2 = nn.Sequential(nn.Linear(16,1),nn.Tanh()) # delta
        self.en3 = nn.Sequential(nn.Linear(16,3),nn.Tanh()) # en
        self.p3 = nn.Sequential(nn.Linear(16,3),nn.ReLU()) # p,每个stage的类别的概率
        self.delta3 = nn.Sequential(nn.Linear(16,1),nn.Tanh()) # delta
    def forward(self,x): # (-1,3,16)
        #idx = torch.tensor([[0,1,2],[0,1,2],[0,1,2]],dtype = torch.float,requires_grad = False)
        idx = torch.zeros(x.size(0),3,dtype = torch.float,requires_grad = False)
        idx[:,0] = 0
        idx[:,1] = 1
        idx[:,2] = 2
        # ()
        en1 = self.en1(x[:,0,:]) # (-1,3) 
        p1 = self.p1(x[:,0:1,:])#(-1,1,3)
        delta1 = self.delta1(x[:,0,:]).squeeze() # (-1,1)
        en2 = self.en2(x[:,1,:])
        p2 = self.p2(x[:,1:2,:])
        delta2 = self.delta2(x[:,1,:]).squeeze()
        en3 = self.en3(x[:,2,:])
        p3 = self.p3(x[:,2:,:])
        delta3 = self.delta3(x[:,2,:]).squeeze()
        result = torch.zeros(x.size(0),3)
        result[:,0] = p1.bmm((idx + en1).view(-1,3,1)).squeeze() * 101 / (1 + delta1) / 3
        result[:,1] = p2.bmm((idx + en2).view(-1,3,1)).squeeze() * 101 / (1 + delta2) / 3 / 3
        result[:,2] = p3.bmm((idx + en3).view(-1,3,1)).squeeze() * 101 / (1 + delta3) / 3 / 3 / 3
        return torch.sum(result,dim = 1)

# for 
class FSA_NET(nn.Module):
    def __init__(self):
        super(FSA_NET,self).__init__()
        self.s1_1 = nn.Sequential(
            BaseBlock(3,16,activation = "relu"),
            nn.AvgPool2d(2)
        )
        self.s1_2 = nn.Sequential(
            BaseBlock(16, 32, activation = "relu"),
            nn.AvgPool2d(2),
            BaseBlock(32, 64, activation="relu"),
            nn.AvgPool2d(2)
        )
        self.s1_3 = nn.Sequential(
            BaseBlock(64, 128),
            BaseBlock(128, 128)
        )
        self.s2_1 = nn.Sequential(
            BaseBlock(3,16,activation = "tanh"),
            nn.MaxPool2d(2)
        )
        self.s2_2 = nn.Sequential(
            BaseBlock(16, 32, activation = "tanh"),
            nn.MaxPool2d(2),
            BaseBlock(32, 64, activation = "tanh"),
            nn.MaxPool2d(2)
        )
        self.s2_3 = nn.Sequential(
            BaseBlock(64, 128),
            BaseBlock(128, 128)
        )
        self.u1 = FeatureExtrator(16,64,4)
        self.u2 = FeatureExtrator(64,64,1)
        self.u3 = FeatureExtrator(128,64,1)

        self.score1 = nn.Conv2d(64,1,kernel_size = 1)
        self.score2 = nn.Conv2d(64,1,kernel_size = 1)
        self.score3 = nn.Conv2d(64,1,kernel_size = 1)
        self.fc1 = nn.Linear(1,15)
        self.fc2 = nn.Linear(1,15)
        self.fc3 = nn.Linear(1,15)
        self.fc4 = nn.Linear(64*3,35)
        self.caps = CapsuleLayer(in_channels = 64)
        self.ssr_result = SSRModule()
    def forward(self,x):
        
        s1_1 = self.s1_1(x)
        s1_2 = self.s1_2(s1_1)
        s1_3 = self.s1_3(s1_2)

        s2_1 = self.s2_1(x)
        s2_2 = self.s2_2(s2_1)
        s2_3 = self.s2_3(s2_2)
        u1 = self.u1(s1_1 * s2_1)
        u2 = self.u2(s1_2 * s2_2)
        u3 = self.u3(s1_3 * s2_3)
        A1,A2,A3 = self.score1(u1),self.score2(u2),self.score3(u3)
        U = torch.cat([u1.view(-1,8,8,64,1),u2.view(-1,8,8,64,1),u3.view(-1,8,8,64,1)],dim = -1).view(-1,64,64,3).permute(0,1,3,2).contiguous().view(-1,64*3,64)
        def A2M(A1,A2,A3,U):# w,h,1 (hw3,c)
            
            M1 = F.sigmoid(self.fc1(A1.view(-1,A1.size(1) * A1.size(2),1))).view(A1.size(0),3*64,5).permute(0,2,1)
            
            M2 = F.sigmoid(self.fc2(A2.view(-1,A2.size(1) * A2.size(2),1))).view(A1.size(0),3*64,5).permute(0,2,1)
            
            M3 = F.sigmoid(self.fc3(A3.view(-1,A3.size(1) * A3.size(2),1))).view(A1.size(0),3*64,5).permute(0,2,1)
            
            A = torch.cat([A1,A2,A3],dim = -1) # (-1,8,8,3)
            C = F.sigmoid(self.fc4(A.view(-1,64*3)).view(-1,7,5)) #(-1,7,5)
            return torch.cat([torch.bmm(C,M1).bmm(U),torch.bmm(C,M2).bmm(U),torch.bmm(C,M3).bmm(U)],dim = 1) # (7*3,c)
        U_new = A2M(A1,A2,A3,U).view(-1,3*7,64) #(-1,21,64)
        U_caps = self.caps(U_new).view(-1,3,16)
        return self.ssr_result(U_caps)
    

if __name__ == "__main__":
    
    net = FSA_NET()
    x = torch.randn(1,3,64,64)
    print(net(x).size())
    from torchsummary import summary
    summary(net,(3,64,64),device = "cpu")
