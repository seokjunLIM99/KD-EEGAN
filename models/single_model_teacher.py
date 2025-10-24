import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
from . import networks_teacher
import sys


class SingleModel_t(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):# 모델 생성 후 초기화 내용 여기에 다 포함되어있음
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)

        self.vgg_loss = networks.PerceptualLoss(opt)
        self.vgg_loss.cuda()
        self.vgg = networks.load_vgg16("./model", self.gpu_ids)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False


        skip = True if opt.skip > 0 else False
        print("Current opt settings:", vars(self.opt))

        self.netG_A = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf,
            which_model_netG='sid_unet_resize',  # teacher 모델 구조
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            gpu_ids=self.gpu_ids,
            skip=skip,
            opt=opt
        )

        #teacher_weight_path = 'D:/checkpoint/enlight/sd300ep_seed42(best)/weight/270_net_G_A.pth'
        teacher_weight_path = opt.teacher_weight
        self.netG_A.load_state_dict(torch.load(teacher_weight_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        for param in self.netG_A.parameters():
            param.requires_grad = False
        self.netG_A.eval()


        print('---------- Teacher Networks load -------------')
        print('----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    


    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)


    def predict(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        A_gray = util.atten2im(self.real_A_gray.data)

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    def get_feature_maps(self):
        """ 모델의 입력 데이터를 사용하여 피처맵을 반환하는 함수 """
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)

        # 네트워크를 통해 피처맵 추출
        if self.opt.skip == 1:
            output, _, enc_first, enc_last, last_representation = self.netG_A.forward(self.real_img, self.real_A_gray)
        else:
            enc_first, enc_last, output, last_representation = None, None, None, None  # skip=False일 경우 피처맵 없음

        return enc_first, enc_last, output, last_representation

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A, enc_first, enc_last, last_representation = self.netG_A.forward(self.real_img, self.real_A_gray)################################# 여기서 Unet_resize_conv 호출하는듯, 모델 forward에서 피처맵 받아오기 ...1
        else:
            self.fake_B = self.netG_A.forward(self.real_img, self.real_A_gray)
        if self.opt.patchD:# True
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

            self.fake_patch = self.fake_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.real_patch = self.real_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.input_patch = self.real_A[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_patch_1.append(self.fake_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])

        # def forward(self):#원본
        #     self.real_A = Variable(self.input_A)
        #     self.real_B = Variable(self.input_B)
        #     self.real_A_gray = Variable(self.input_A_gray)
        #     self.real_img = Variable(self.input_img)
        #     if self.opt.noise > 0:
        #         self.noise = Variable(
        #             torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise / 255.))
        #         self.real_A = self.real_A + self.noise
        #     if self.opt.input_linear:
        #         self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))
        #     if self.opt.skip == 1:
        #         self.fake_B, self.latent_real_A  = self.netG_A.forward(self.real_img,
        #                                                                                    self.real_A_gray)  ################################# 여기서 Unet_resize_conv 호출하는듯, 모델 forward에서 피처맵 받아오기 ...1
        #     else:
        #         self.fake_B = self.netG_A.forward(self.real_img, self.real_A_gray)
        #     if self.opt.patchD:  # True
        #         w = self.real_A.size(3)
        #         h = self.real_A.size(2)
        #         w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
        #         h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))
        #
        #         self.fake_patch = self.fake_B[:, :, h_offset:h_offset + self.opt.patchSize,
        #                           w_offset:w_offset + self.opt.patchSize]
        #         self.real_patch = self.real_B[:, :, h_offset:h_offset + self.opt.patchSize,
        #                           w_offset:w_offset + self.opt.patchSize]
        #         self.input_patch = self.real_A[:, :, h_offset:h_offset + self.opt.patchSize,
        #                            w_offset:w_offset + self.opt.patchSize]
        #     if self.opt.patchD_3 > 0:
        #         self.fake_patch_1 = []
        #         self.real_patch_1 = []
        #         self.input_patch_1 = []
        #         w = self.real_A.size(3)
        #         h = self.real_A.size(2)
        #         for i in range(self.opt.patchD_3):
        #             w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
        #             h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
        #             self.fake_patch_1.append(self.fake_B[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #                                      w_offset_1:w_offset_1 + self.opt.patchSize])
        #             self.real_patch_1.append(self.real_B[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #                                      w_offset_1:w_offset_1 + self.opt.patchSize])
        #             self.input_patch_1.append(self.real_A[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #                                       w_offset_1:w_offset_1 + self.opt.patchSize])

    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        if not self.opt.patchD:
            self.optimizer_D_A.step()
        else:
            # self.forward()
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            self.optimizer_D_A.step()
            self.optimizer_D_P.step()

    def get_current_errors(self, epoch):
        # ✅ .data[0] → .item()으로 수정
        D_A = self.loss_D_A.item()
        D_P = self.loss_D_P.item() if self.opt.patchD else 0
        G_A = self.loss_G_A.item()

        if self.opt.vgg > 0:
            vgg = self.loss_vgg_b.item() / self.opt.vgg if self.opt.vgg > 0 else 0
            return OrderedDict([
                ('D_A', D_A),
                ('G_A', G_A),
                ("vgg", vgg),
                ("D_P", D_P)
            ])
        elif self.opt.fcn > 0:
            fcn = self.loss_fcn_b.item() / self.opt.fcn if self.opt.fcn > 0 else 0
            return OrderedDict([
                ('D_A', D_A),
                ('G_A', G_A),
                ("fcn", fcn),
                ("D_P", D_P)
            ])
        

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            latent_show = util.latent2im(self.latent_real_A.data)
            if self.opt.patchD:
                fake_patch = util.tensor2im(self.fake_patch.data)
                real_patch = util.tensor2im(self.real_patch.data)
                if self.opt.patch_vgg:
                    input_patch = util.tensor2im(self.input_patch.data)
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch), ('self_attention', self_attention)])
                else:
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('self_attention', self_attention)])
            else:
                if not self.opt.self_attention:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B)])
                else:
                    self_attention = util.atten2im(self.real_A_gray.data)
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('latent_real_A', latent_real_A), ('latent_show', latent_show),
                                    ('self_attention', self_attention)])
        else:
            if not self.opt.self_attention:
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
            else:
                self_attention = util.atten2im(self.real_A_gray.data)
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('self_attention', self_attention)])