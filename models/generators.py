import torch
import torch.nn as nn

from . import common
# import common

class TransferNet(nn.Module):
    def __init__(self, n_feat=48, z_feat=12, leaky_neg=0.2, n_resblock=6, bn=True, rgb_range=255, rgb_mean=(0.5, 0.5, 0.5), in_chan=3):
        super(TransferNet, self).__init__()
        print("n_feat: ", n_feat)
        print("z_feat: ", z_feat)
        rgb_std = (1.0, 1.0, 1.0)

        if in_chan == 1:
            rgb_std = torch.Tensor((rgb_std[0], ))
            rgb_mean = torch.Tensor((rgb_mean[0], ))

        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, in_chan=in_chan)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1, in_chan=in_chan)

        leaky_neg = leaky_neg
        filter_size = 5
        z_channel = z_feat
        in_img = [common.default_conv(in_chan, n_feat//2, filter_size)]
        if bn:
            in_img.append(nn.BatchNorm2d(n_feat//2))
        in_img.append(nn.LeakyReLU(leaky_neg))
        in_img.append(common.ResBlock(common.default_conv, n_feat//2, filter_size, bn=bn, act=nn.LeakyReLU(leaky_neg)))
        self.img_head = nn.Sequential(*in_img)

        in_z = [nn.ConvTranspose2d(1, z_channel, 2, 2, 0, 0), # 8 -> 16
                nn.LeakyReLU(leaky_neg),
                nn.ConvTranspose2d(z_channel, 2 * z_channel, 1, 1, 0, 0), # 16 -> 32
                nn.LeakyReLU(leaky_neg)]
        self.z_head = nn.Sequential(*in_z)
        self.merge = nn.Conv2d(n_feat//2 + 2*z_channel, n_feat, 1, 1, 0)
        resblocks = [
            common.ResBlock(common.default_conv, n_feat, filter_size, bn=bn, act=nn.LeakyReLU(leaky_neg)) \
            for _ in range(n_resblock)]
        self.res_blocks = nn.Sequential(*resblocks)
        self.fusion = nn.Sequential(
            common.default_conv(n_feat, n_feat//2, 1),
            nn.LeakyReLU(leaky_neg),
            common.default_conv(n_feat//2, n_feat//4, 1),
            nn.LeakyReLU(leaky_neg),
            common.default_conv(n_feat//4, in_chan, 1))

    def forward(self, x, z=None):

        # print("raw_x: ", x.shape)
        # print("raw_z: ", z.shape)
        out_x = self.sub_mean(x)
        # print("sub_mean: ", out_x.shape)
        out_x = self.img_head(out_x)
        # print("out_x: ", out_x.shape)
        out_z = self.z_head(z)
        # print("out_z: ", out_z.shape)
        out = self.merge(torch.cat((out_x, out_z), dim=1))
        out = self.res_blocks(out)
        out = self.fusion(out)

        out = self.add_mean(out)
        # exit()
        return out

if __name__ == "__main__":
    rgb_range = 1
    rgb_mean = (0.0, 0.0, 0.0)
    model = TransferNet(n_feat=48, z_feat=12, n_resblock=5, bn=False, rgb_range=rgb_range, rgb_mean=rgb_mean, in_chan=1)
    print(model)

    X = torch.rand(2, 1, 48, 48, dtype=torch.float32) * rgb_range
    Z = torch.randn(2, 1, 12, 12, dtype=torch.float32)
    Y = model(X, Z).detach()
    print(X.shape, Y.shape)
    print(Y.min(), Y.max())
