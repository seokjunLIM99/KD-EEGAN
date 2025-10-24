import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from models.models_teacher import create_model_teacher
from tqdm import tqdm
import torch.multiprocessing as mp
import numpy as np
import torch

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def train():

    opt = TrainOptions().parse()
    config = get_config(opt.config)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    student_model = create_model(opt)
    teacher_model = create_model_teacher(opt)

    total_steps = 0
    weight_dir = r"D:\checkpoint\weights"

    for epoch in tqdm(range(1, opt.niter + opt.niter_decay + 1), desc="Training Progress", unit="epoch"):
        epoch_start_time = time.time()

        for i, data in enumerate(tqdm(dataset, desc=f"Epoch {epoch}", leave=False, unit="batch")):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)

            student_model.set_input(data)
            teacher_model.set_input(data)

            student_feature_first, student_feature_last, student_output, student_last = student_model.get_feature_maps()
            with torch.no_grad():
                teacher_feature_first, teacher_feature_last, teacher_output, teacher_last = teacher_model.get_feature_maps()


            teacher_feature_last_spp_at = student_model.spp_at_t(teacher_feature_last)
            student_feature_last_spp_at = student_model.spp_at_s(student_feature_last)

            enc_first_loss = torch.nn.functional.mse_loss(student_feature_first, teacher_feature_first)
            enc_last_loss= torch.nn.functional.mse_loss(student_feature_last_spp_at, teacher_feature_last_spp_at)

            kd_loss = enc_first_loss + enc_last_loss

            student_model.optimize_parameters(epoch, kd_loss)

            if total_steps % opt.print_freq == 0:
                errors = student_model.get_current_errors(epoch)
                t = (time.time() - iter_start_time) / opt.batchSize
                tqdm.write(f"(Epoch: {epoch}, Iter: {epoch_iter}, Time: {t:.3f}) Errors: {errors}")

        if epoch % opt.save_epoch_freq == 0:
            tqdm.write(f"Saving the model at the end of epoch {epoch}, iters {total_steps}")
            student_model.save('latest', weight_dir)
            student_model.save(epoch, weight_dir)

        tqdm.write(
            f"End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec")

        if opt.new_lr:
            if epoch in [opt.niter, opt.niter + 20, opt.niter + 70, opt.niter + 90]:
                student_model.update_learning_rate()
                if epoch == opt.niter + 90:
                    for _ in range(4): student_model.update_learning_rate()
        else:
            if epoch > opt.niter:
                student_model.update_learning_rate()

if __name__ == '__main__':
    mp.freeze_support()
    train()
