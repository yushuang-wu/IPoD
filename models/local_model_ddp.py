import torch
import torch.nn as nn
import torch.nn.functional as F

# Functions
##############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DiscriminatorFC(nn.Module):
    def __init__(self, latent_dim=128, n_features=(256, 512)):
        super(DiscriminatorFC, self).__init__()
        self.n_features = list(n_features)

        model = []
        prev_nf = latent_dim
        for idx, nf in enumerate(self.n_features):
            model.append(nn.Linear(prev_nf, nf))
            model.append(nn.LeakyReLU(inplace=True))
            prev_nf = nf

        model.append(nn.Linear(self.n_features[-1], 1))

        self.model = nn.Sequential(*model)

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.model(x)#.view(-1)
        return x

# ShapeNet Pointcloud Completion ---------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

class ShapeNetPoints(nn.Module):

    def __init__(self, hidden_dim=256):
        super(ShapeNetPoints, self).__init__()
        # 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='zeros')
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='zeros')
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='zeros')
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='zeros')
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        # self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        # self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        # self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        # self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)
        self.register_buffer('displacments', torch.Tensor(displacments)) # TODO add register_buffer for ddp
        # self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p, padding_mode='zeros')  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='zeros')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='zeros')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='zeros')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='zeros')
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='zeros')

        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5), dim=1)  # (B, features, 1, 7, sample_num)

        return features

class ScanNetPoints(nn.Module):

    def __init__(self, hidden_dim=256):
        super(ScanNetPoints, self).__init__()
        # 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='zeros')
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='zeros')
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='zeros')
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='zeros')
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)
        self.register_buffer('displacments', torch.Tensor(displacments)) # TODO add register_buffer for ddp
        # self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x, features_sp):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feat_sc_0 = F.grid_sample(x, p, padding_mode='zeros')  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feat_sc_1 = F.grid_sample(net, p, padding_mode='zeros')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feat_sc_2 = F.grid_sample(net, p, padding_mode='zeros')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feat_sc_3 = F.grid_sample(net, p, padding_mode='zeros')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feat_sc_4 = F.grid_sample(net, p, padding_mode='zeros')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feat_sc_5 = F.grid_sample(net, p, padding_mode='zeros')  # out : (B,C (of x), 1,1,sample_num)
        
        features_sc = torch.cat((feat_sc_0, feat_sc_1, feat_sc_2, feat_sc_3, feat_sc_4, feat_sc_5), dim=1)

        b, c0, c1, c2, c3, c4, c5 = feat_sc_0.shape[0], feat_sc_0.shape[1], feat_sc_1.shape[1], feat_sc_2.shape[1], feat_sc_3.shape[1], feat_sc_4.shape[1], feat_sc_5.shape[1]
        weights = []
        for i, c in enumerate([c0, c1, c2, c3, c4, c5]):
            if i < 2:
                ratio = 0.
            elif i < 4:
                ratio = 0.5
            else:
                ratio = 1.
            # ratio = i/5.
            weights_i = torch.ones((b, c)) * ratio
            weights += [weights_i.cuda()]
        weights_sp = torch.cat(weights, dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4).cuda()
        weights_sc = 1 - weights_sp
        features = features_sc * weights_sc + features_sp * weights_sp

        return features #out


class ImplicitFunction(nn.Module):

    def __init__(self, hidden_dim=256):
        super(ImplicitFunction, self).__init__()

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        # feature_size = (16 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

    def forward(self, features):

        shape = features.shape
        features = torch.reshape(features, (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out

