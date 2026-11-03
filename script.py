import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Run KD-EEGAN training.')
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--gpu_ids', default='0')
    args = parser.parse_args()

    command = [
        sys.executable, 'train.py',
        '--dataroot', args.dataroot,
        '--gpu_ids', args.gpu_ids,
        '--name', 'kd-eegan',
        '--no_dropout',
        '--patchD',
        '--patch_vgg',
        '--self_attention',
        '--use_ragan',
        '--hybrid_loss',
        '--times_residual',
    ]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()

