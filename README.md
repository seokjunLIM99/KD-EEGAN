# KD-EEGAN

Place paired images with matching filenames in `<dataset>/trainA` (degraded) and `<dataset>/trainB` (clear). Put the pretrained files at `weights/teacher.pth`, `weights/yolov9-smoke.pt`, and `weights/vgg16.weight`.

## Train

```bash
python train.py
```
