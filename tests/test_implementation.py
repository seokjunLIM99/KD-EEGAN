import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from data.base_dataset import get_params, get_transform
from models.networks import (
    Unet_resize_conv,
    Unet_resize_conv_student,
    complex_frequency_residual,
    pixel_reconstruction_l1,
    spatial_feature_mse,
)
from models.single_model import CBAM, SpatialPyramidPoolingWithAttention
from options.train_options import TrainOptions


class ImplementationTests(unittest.TestCase):
    def test_hyperparameter_defaults(self):
        options = TrainOptions()
        options.initialize()
        opt = options.parser.parse_args(['--dataroot', 'dataset'])
        self.assertEqual(opt.batchSize, 8)
        self.assertEqual((opt.niter, opt.niter_decay), (150, 150))
        self.assertEqual((opt.lr, opt.beta1), (1e-4, 0.9))
        self.assertEqual(opt.patchD_3 + 1, 5)
        self.assertEqual((opt.skip, opt.vgg_choose), (1, 'relu5_1'))
        self.assertEqual(
            (opt.lambda_spatial, opt.lambda_fft, opt.lambda_pixel),
            (0.2, 0.2, 1.0))
        self.assertEqual(
            (opt.lambda_kd_shallow, opt.lambda_kd_deep, opt.lambda_student),
            (0.5, 0.5, 1.0))

    def test_generator_parameter_counts(self):
        opt = SimpleNamespace(skip=1)
        teacher = Unet_resize_conv(opt, skip=True)
        student = Unet_resize_conv_student(opt, skip=True)
        teacher_count = sum(p.numel() for p in teacher.parameters())
        student_count = sum(p.numel() for p in student.parameters())
        self.assertEqual(teacher_count, 8_636_675)
        self.assertEqual(student_count, 4_504_579)

    def test_teacher_and_student_forward_scales(self):
        opt = SimpleNamespace(skip=1)
        image = torch.randn(1, 3, 64, 64)
        illumination = torch.rand(1, 1, 64, 64)
        with torch.no_grad():
            teacher = Unet_resize_conv(opt, skip=True)(image, illumination)
            student = Unet_resize_conv_student(opt, skip=True)(image, illumination)
        self.assertEqual(teacher[0].shape, image.shape)
        self.assertEqual(student[0].shape, image.shape)
        self.assertEqual(teacher[3].shape, (1, 512, 4, 4))
        self.assertEqual(student[3].shape, (1, 512, 8, 8))

    def test_dsfl_losses(self):
        fake = torch.ones(1, 3, 2, 2)
        real = torch.zeros_like(fake)
        self.assertAlmostEqual(spatial_feature_mse(fake, real).item(), 1.0)
        self.assertAlmostEqual(complex_frequency_residual(fake, real).item(), 1.0)
        self.assertAlmostEqual(pixel_reconstruction_l1(fake, real).item(), 1.0)

    def test_at_spp_shape_and_non_residual_attention(self):
        torch.manual_seed(0)
        x = torch.ones(1, 512, 20, 20)
        attention = CBAM(channels=512, r=16)
        attended = attention(x)
        self.assertLessEqual(attended.max().item(), 1.0)
        spp = SpatialPyramidPoolingWithAttention(channels=512)
        self.assertEqual(spp(x).shape, (1, 512 * (1 + 25 + 100 + 400)))

    def test_paired_transform_uses_identical_crop_and_flip(self):
        image = Image.fromarray(
            torch.arange(3 * 24 * 24, dtype=torch.uint8)
            .reshape(24, 24, 3).numpy())
        opt = SimpleNamespace(
            resize_or_crop='crop', fineSize=16, loadSize=16,
            isTrain=True, no_flip=False)
        params = get_params(opt, image.size)
        transform = get_transform(opt, params)
        self.assertTrue(torch.equal(transform(image), transform(image.copy())))


if __name__ == '__main__':
    unittest.main()
