import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device(
            'cuda:%d' % opt.gpu_ids[0] if opt.gpu_ids else 'cpu')
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

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

    def save_network(self, network, network_label, epoch_label, gpu_ids, save_dir):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)

        os.makedirs(save_dir, exist_ok=True)
        state_dict = network.cpu().state_dict()
        torch.save(state_dict, save_path)

        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    def load_network(self, network, network_label, epoch_label, load_dir=None):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(load_dir or self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path, map_location=self.device))

    def update_learning_rate():
        pass
