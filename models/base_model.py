import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids, save_dir):# att_spp 관련 파라미터 제외 후 저장
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)

        # ✅ 디렉토리 생성 (존재하지 않으면 생성)
        os.makedirs(save_dir, exist_ok=True)

        # ✅ CBAM 파라미터 제거
        state_dict = network.cpu().state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if "spp_at" not in k}  # CBAM 키 제거

        # ✅ 필터링된 가중치 저장
        torch.save(filtered_state_dict, save_path)

        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])


    # def save_network(self, network, network_label, epoch_label, gpu_ids, save_dir):# 원본 sav_network
    #     save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    #     save_path = os.path.join(save_dir, save_filename)
    #
    #     # ✅ 디렉토리 생성 (존재하지 않으면 생성)
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     torch.save(network.cpu().state_dict(), save_path)
    #     if len(gpu_ids) and torch.cuda.is_available():
    #         network.cuda(device=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        pass
