import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from models.rcan import make_cleaning_net, make_SR_net
from models.rrdb import RRDBNet
from models.generators import TransferNet
from models.vggs import VGGFeatureExtractor
from models.discriminators import NLayerDiscriminator
from models.losses import GANLoss
from models.geo_loss import geometry_ensemble
from models.denoising_diffusion_pytorch import Unet, GaussianDiffusion

from collections import OrderedDict
from einops import rearrange, reduce

# from rcan import make_cleaning_net, make_SR_net
# from rrdb import RRDBNet
# from generators import TransferNet
# from vggs import VGGFeatureExtractor
# from discriminators import NLayerDiscriminator
# from losses import GANLoss
# from geo_loss import geometry_ensemble
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion

class Galaxy_Diffusion_Model():
    def __init__(self, device, cfg, use_ddp=False):
        self.device = device
        self.use_esrgan = cfg.SR.MODEL == "ESRGAN"
        self.sr_warmup_iter = cfg.OPT_SR.WARMUP
        self.idt_input_clean = cfg.OPT_CYC.IDT_INPUT == "clean"
        rgb_range = cfg.DATA.IMG_RANGE
        rgb_mean_point = (0.5, 0.5, 0.5) if cfg.DATA.IMG_MEAN_SHIFT else (0, 0, 0)
        self.in_chan = cfg.DATA.CHANNEL
        
        self.G_xy = make_cleaning_net(rgb_range=rgb_range, rgb_mean=rgb_mean_point, in_chan=self.in_chan).to(device)
        self.G_yx = TransferNet(rgb_range=rgb_range, rgb_mean=rgb_mean_point, in_chan=self.in_chan).to(device)
        
        unet = Unet(dim = 128, init_dim=128, dim_mults = (1, 4, 8), channels=1, resnet_block_groups = 4).to(device)
        self.diffusion = GaussianDiffusion(unet, image_size = 12, timesteps = 1000, loss_type = 'l2', objective='pred_x0').to(device)
        
        self.U = RRDBNet(self.in_chan, self.in_chan, scale_factor=cfg.SR.SCALE).to(device)
        
        self.D_x = NLayerDiscriminator(self.in_chan, scale_factor=1, norm_layer=nn.Identity).to(device)
        self.D_y = NLayerDiscriminator(self.in_chan, scale_factor=1, norm_layer=nn.Identity).to(device)
        self.D_sr = NLayerDiscriminator(self.in_chan, scale_factor=cfg.SR.SCALE, norm_layer=nn.Identity).to(device)
        self.D_esrgan = NLayerDiscriminator(self.in_chan, scale_factor=1, norm_layer=nn.InstanceNorm2d).to(device)
        
        if use_ddp:
            self.G_xy = DDP(self.G_xy, device_ids=[device])
            self.G_yx = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.G_yx), device_ids=[device])
            unet = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet), device_ids=[device])
            self.diffusion = DDP(torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.diffusion), device_ids=[device])
            self.U = DDP(self.U, device_ids=[device])
            self.D_x = DDP(self.D_x, device_ids=[device])
            self.D_y = DDP(self.D_y, device_ids=[device])
            self.D_sr = DDP(self.D_sr, device_ids=[device])
            self.D_esrgan = DDP(self.D_esrgan, device_ids=[device])

        self.opt_Gxy = optim.Adam(self.G_xy.parameters(), lr=cfg.OPT_CYC.LR_G, betas=cfg.OPT_CYC.BETAS_G)
        self.opt_Gyx = optim.Adam(self.G_yx.parameters(), lr=cfg.OPT_CYC.LR_G, betas=cfg.OPT_CYC.BETAS_G)
        self.opt_diffusion = optim.Adam(self.diffusion.parameters(), lr=cfg.OPT_CYC.LR_G, betas=cfg.OPT_CYC.BETAS_G)
        self.opt_Dx = optim.Adam(self.D_x.parameters(), lr=cfg.OPT_CYC.LR_D, betas=cfg.OPT_CYC.BETAS_D)
        self.opt_Dy = optim.Adam(self.D_y.parameters(), lr=cfg.OPT_CYC.LR_D, betas=cfg.OPT_CYC.BETAS_D)
        self.opt_Dsr = optim.Adam(self.D_sr.parameters(), lr=cfg.OPT_CYC.LR_D, betas=cfg.OPT_CYC.BETAS_D)
        self.opt_U = optim.Adam(self.U.parameters(), lr=cfg.OPT_SR.LR_G, betas=cfg.OPT_SR.BETAS_G)
        self.opt_D_esrgan = optim.Adam(self.D_esrgan.parameters(), lr=cfg.OPT_SR.LR_D, betas=cfg.OPT_SR.BETAS_D)
        
        self.lr_Gxy = optim.lr_scheduler.MultiStepLR(self.opt_Gxy, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Gyx = optim.lr_scheduler.MultiStepLR(self.opt_Gyx, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_diffusion = optim.lr_scheduler.MultiStepLR(self.opt_diffusion, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Dx = optim.lr_scheduler.MultiStepLR(self.opt_Dx, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Dy = optim.lr_scheduler.MultiStepLR(self.opt_Dy, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Dsr = optim.lr_scheduler.MultiStepLR(self.opt_Dsr, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_U = optim.lr_scheduler.MultiStepLR(self.opt_U, milestones=cfg.OPT_SR.LR_MILESTONE, gamma=cfg.OPT_SR.LR_DECAY)
        self.lr_D_esrgan = optim.lr_scheduler.MultiStepLR(self.opt_D_esrgan, milestones=cfg.OPT_SR.LR_MILESTONE, gamma=cfg.OPT_SR.LR_DECAY)
        
        self.nets = {"G_xy":self.G_xy, "G_yx":self.G_yx, "U":self.U, "D_x":self.D_x, "D_y":self.D_y, "D_sr":self.D_sr, "D_esrgan": self.D_esrgan, "diffusion": self.diffusion}
        self.optims = {"G_xy":self.opt_Gxy, "G_yx":self.opt_Gyx, "U":self.opt_U, "D_x":self.opt_Dx, "D_y":self.opt_Dy, "D_sr":self.opt_Dsr, "D_esrgan": self.opt_D_esrgan, "diffusion": self.opt_diffusion}
        self.lr_decays = {"G_xy":self.lr_Gxy, "G_yx":self.lr_Gyx, "U":self.lr_U, "D_x":self.lr_Dx, "D_y":self.lr_Dy, "D_sr":self.lr_Dsr, "D_esrgan": self.lr_D_esrgan, "diffusion": self.lr_diffusion}
        self.discs = ["D_x", "D_y", "D_sr", "D_esrgan"]
        self.gens = ["G_xy", "G_yx", "U", "diffusion"]

        self.n_iter = 0
        self.gan_loss = GANLoss("lsgan")
        self.l1_loss = nn.L1Loss()

        self.d_sr_weight = cfg.OPT_CYC.LOSS.D_SR_WEIGHT
        self.cyc_weight = cfg.OPT_CYC.LOSS.CYC_WEIGHT
        self.idt_weight = cfg.OPT_CYC.LOSS.IDT_WEIGHT
        self.geo_weight = cfg.OPT_CYC.LOSS.GEO_WEIGHT
    
        self.ragan_loss = GANLoss("vanilla")
        self.vgg_feat = "conv5_4"
        self.vgg = VGGFeatureExtractor([self.vgg_feat], use_input_norm=True, range_norm=False).to(device)

        self.sr_pix_weight = cfg.OPT_SR.LOSS.PIXEL_WEIGHT
        self.sr_vgg_weight = cfg.OPT_SR.LOSS.VGG_WEIGHT
        self.sr_gan_weight = cfg.OPT_SR.LOSS.GAN_WEIGHT



    def net_save(self, folder, shout=False):
        file_name = os.path.join(folder, f"nets_{self.n_iter}.pth")
        nets = {k:v.state_dict() for k, v in self.nets.items()}

        optims = {k:v.state_dict() for k, v in self.optims.items()}
        lr_decays = {k:v.state_dict() for k, v in self.lr_decays.items()}
        alls = {"nets":nets, "optims":optims, "lr_decays":lr_decays}
        torch.save(alls, file_name)
        if shout: print("Saved: ", file_name)
        return file_name

    def net_load(self, file_name, strict=True):
        map_loc = {"cuda:0": f"cuda:{self.device}"}

        map_loc = 'cuda'
        loaded = torch.load(file_name, map_location=map_loc)

        for n in self.nets:
            for key in list(loaded["nets"][n].keys()):
                if 'module.' in key:
                    loaded["nets"][n][key.replace('module.', '')] = loaded["nets"][n][key]
                    del loaded["nets"][n][key]
            self.nets[n].load_state_dict(loaded["nets"][n], strict=strict)
            
        for o in self.optims:
            self.optims[o].load_state_dict(loaded["optims"][o])
        for l in self.lr_decays:
            self.lr_decays[l].load_state_dict(loaded["lr_decays"][l])

    def net_grad_toggle(self, nets, need_grad):
        for n in nets:
            for p in self.nets[n].parameters():
                p.requires_grad = need_grad

    def mode_selector(self, mode="train"):
        if mode == "train":
            for n in self.nets:
                self.nets[n].train()
        elif mode in ["eval", "test"]:
            for n in self.nets:
                self.nets[n].eval()
                
    def warmup_checker(self):
        return self.n_iter <= self.sr_warmup_iter
    
    def lr_decay_step(self, shout=False):
        lrs = "\nLearning rates: "
        changed = False
        for i, n in enumerate(self.lr_decays):
            if self.warmup_checker() and n == "D_esrgan":
                continue
            lr_old = self.lr_decays[n].get_last_lr()[0]
            self.lr_decays[n].step()
            lr_new = self.lr_decays[n].get_last_lr()[0]
            if lr_old != lr_new:
                changed = True
                lrs += f", {n}={self.lr_decays[n].get_last_lr()[0]}" if i > 0 else f"{n}={self.lr_decays[n].get_last_lr()[0]}"
        if shout and changed: print(lrs)

    def extract(a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def test_sample(self, Xs, Yds=None, Zs=None):
        x = None
        with torch.no_grad():
            rec_y, _, y_noised = self.diffusion(Yds, pair_flow=True)
            y = self.diffusion(Xs, pair_flow=False)
            sr = self.U(y)
            
        # if Yds is not None and Zs is not None:
        #     x = self.nets["G_yx"](Yds, Zs)
        return y, sr, rec_y, y_noised

    def train_step(self, Ys, Xs, Yds, Zs):
        '''
        Ys: high resolutions
        Xs: low resolutions
        Yds: down sampled HR
        Zs: noises
        '''
        self.n_iter += 1
        loss_dict = dict()
        # print("Hr: {}, Lr: {}, Dwon Hr: {}, Noise: {}".format(Ys.shape, Xs.shape, Yds.shape, Zs.shape))
        # forward
        # fake_Xs = self.G_yx(Yds, Zs)
        # rec_Yds = self.G_xy(fake_Xs)
        rec_Yds, diffusion_loss, _ = self.diffusion(Yds, pair_flow=True)
        fake_Yds = self.diffusion(Xs, pair_flow=False)
        # fake_Yds = self.G_xy(Xs)
        # geo_Yds = geometry_ensemble(self.G_xy, Xs)
        # idt_out = self.G_xy(Yds) if self.idt_input_clean else fake_Yds
        # idt_out = self.G_xy(Yds) if self.idt_input_clean else fake_Yds
        sr_y = self.U(rec_Yds)
        sr_x = self.U(fake_Yds)

        self.net_grad_toggle(["D_x", "D_y", "D_sr"], True)
        # D_x
        # pred_fake_Xs = self.D_x(fake_Xs.detach())
        # pred_real_Xs = self.D_x(Xs)
        # loss_D_x = (self.gan_loss(pred_real_Xs, True, True) + self.gan_loss(pred_fake_Xs, False, True)) * 0.5
        # self.opt_Dx.zero_grad()
        # loss_D_x.backward()
        # self.opt_Dx.step()
        # loss_dict["D_x"] = loss_D_x.item()

        # D_y ("clean LR from Dirty LR" vs "Down HR")
        pred_fake_Yds = self.D_y(fake_Yds.detach())
        pred_real_Yds = self.D_y(Yds)
        loss_D_y = (self.gan_loss(pred_real_Yds, True, True) + self.gan_loss(pred_fake_Yds, False, True)) * 0.5
        self.opt_Dy.zero_grad()
        loss_D_y.backward()
        self.opt_Dy.step()
        loss_dict["D_y"] = loss_D_y.item()

        # D_sr
        pred_sr_x = self.D_sr(sr_x.detach())
        pred_sr_y = self.D_sr(sr_y.detach())
        loss_D_sr = (self.gan_loss(pred_sr_x, True, True) + self.gan_loss(pred_sr_y, False, True)) * 0.5
        self.opt_Dsr.zero_grad()
        loss_D_sr.backward()
        self.opt_Dsr.step()
        loss_dict["D_sr"] = loss_D_sr.item()

        self.net_grad_toggle(["D_x", "D_y", "D_sr"], False)
        # G_yx
        # self.opt_Gyx.zero_grad()
        # self.opt_Gxy.zero_grad()
        # pred_fake_Xs = self.D_x(fake_Xs)
        # loss_gan_Gyx = self.gan_loss(pred_fake_Xs, True, False)
        # loss_dict["G_yx_gan"] = loss_gan_Gyx.item()

        # G_xy
        pred_fake_Yds = self.D_y(fake_Yds)
        pred_sr_y = self.D_sr(sr_y)
        loss_gan_Gxy = self.gan_loss(pred_fake_Yds, True, False)
        # loss_idt_Gxy = self.l1_loss(idt_out, Yds) if self.idt_input_clean else self.l1_loss(idt_out, Xs)
        loss_cycle = self.l1_loss(rec_Yds, Yds)
        # loss_geo = self.l1_loss(fake_Yds, geo_Yds)
        loss_d_sr = self.gan_loss(pred_sr_y, True, False)
        loss_total_gen = loss_gan_Gxy + self.cyc_weight * loss_cycle + self.idt_weight * diffusion_loss  + self.d_sr_weight * loss_d_sr
        # # loss_total_gen = loss_gan_Gyx + loss_gan_Gxy + self.cyc_weight * loss_cycle + self.idt_weight * loss_idt_Gxy + self.geo_weight * loss_geo + self.d_sr_weight * loss_d_sr
        # loss_total_gen = loss_gan_Gxy + self.cyc_weight * loss_cycle + self.idt_weight * loss_idt_Gxy + self.geo_weight * loss_geo + self.d_sr_weight * loss_d_sr
        loss_dict["G_xy_gan"] = loss_gan_Gxy.item()
        # loss_dict["G_xy_idt"] = loss_idt_Gxy.item()
        loss_dict["cyc_loss"] = loss_cycle.item()
        # loss_dict["G_xy_geo"] = loss_geo.item()
        loss_dict["D_sr"] = loss_d_sr.item()
        loss_dict["Diffusion"] = diffusion_loss.item()
        loss_dict["G_total"] = loss_total_gen.item()
        # Diffusion Loss About HR
        # gen loss backward and update
        self.opt_diffusion.zero_grad()
        loss_total_gen.backward()
        self.opt_diffusion.step()
        # self.opt_Gyx.step()
        # self.opt_Gxy.step()

        # U
        if self.use_esrgan and not self.warmup_checker():
            fake_sr = self.U(rec_Yds.detach())

            # D
            self.net_grad_toggle(["D_esrgan"], True)
            self.opt_D_esrgan.zero_grad()
            fake_pred = self.D_esrgan(fake_sr).detach()
            real_pred = self.D_esrgan(Ys)
            real_loss = self.ragan_loss(real_pred - torch.mean(fake_pred), True, is_disc=True) * 0.5
            real_loss.backward()

            fake_pred = self.D_esrgan(fake_sr.detach())
            fake_loss = self.ragan_loss(fake_pred - torch.mean(real_pred.detach()), False, is_disc=True) * 0.5
            fake_loss.backward()
            self.opt_D_esrgan.step()
            loss_dict["D_esrgan"] = real_loss.item() + fake_loss.item()

            # G
            self.net_grad_toggle(["D_esrgan"], False)
            self.opt_U.zero_grad()
            loss_pix = self.l1_loss(fake_sr, Ys)
            loss_vgg = self.l1_loss(self.vgg(fake_sr)[self.vgg_feat], self.vgg(Ys)[self.vgg_feat].detach())

            real_pred = self.D_esrgan(Ys).detach()
            fake_pred = self.D_esrgan(fake_sr)
            real_loss = self.ragan_loss(real_pred - torch.mean(fake_pred), False, is_disc=False)
            fake_loss = self.ragan_loss(fake_pred - torch.mean(real_pred), True, is_disc=False)
            loss_gan = (real_loss + fake_loss) * 0.5
            loss_U = self.sr_pix_weight * loss_pix + self.sr_vgg_weight * loss_vgg + self.sr_gan_weight * loss_gan
            loss_U.backward()
            self.opt_U.step()
            
            loss_dict["U_pix"] = loss_pix.item()
            loss_dict["U_vgg"] = loss_vgg.item()
            loss_dict["U_gan"] = loss_gan.item()
            loss_dict["U_total"] = loss_U.item()
        else:
            self.opt_U.zero_grad()
            # print(rec_Yds.shape, self.U(rec_Yds.detach()).shape, Ys.shape)
            loss_U = self.l1_loss(self.U(rec_Yds.detach()), Ys)
            loss_U.backward()
            self.opt_U.step()
            loss_dict["U_pix"] = loss_U.item()
            
        return loss_dict

if __name__ == "__main__":
    from yacs.config import CfgNode
    with open("/home/s20225004/pseudo-sr/configs/GalaxyZoo.yaml", "rb") as cf:
        CFG = CfgNode.load_cfg(cf)
        CFG.freeze()
        
    print(CFG)
    device = 0
    x = torch.randn(8, CFG.DATA.CHANNEL, 12, 12, dtype=torch.float32, device=device)
    y = torch.randn(8, CFG.DATA.CHANNEL, 48, 48, dtype=torch.float32, device=device)
    yd = torch.randn(8, CFG.DATA.CHANNEL, 12, 12, dtype=torch.float32, device=device)
    z = torch.randn(8, 1, 6, 6, dtype=torch.float32, device=device)
    model = Galaxy_Diffusion_Model(device, CFG)
    losses = model.train_step(y, x, yd, z)
    file_name = model.net_save(".", True)
    model.net_load(file_name)
    for i in range(110000):
        model.lr_decay_step(True)
    info = f"  1/(1):"
    for i, itm in enumerate(losses.items()):
        info += f", {itm[0]}={itm[1]:.3f}" if i > 0 else f" {itm[0]}={itm[1]:.3f}"
    print(info)
    print("fin")
